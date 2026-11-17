# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from types import SimpleNamespace

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
    LwdProbeSubmission,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdPrefixManifest, LwdProbeResult


def _connection(edge_id: int, cloud_id: int, dp_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        connection_id=LwdConnectionId(edge_id, cloud_id, dp_idx),
        cloud=SimpleNamespace(
            coord_url=f"http://cloud{cloud_id}:{8100 + dp_idx}",
            coord_port=8100 + dp_idx,
            addr=f"cloud{cloud_id}",
        ),
    )


def _config(prefix_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            connections=[_connection(0, 0, 0), _connection(0, 0, 1)],
            coordination=SimpleNamespace(
                enabled=True,
                placement_only=False,
                consumer_id=None,
                connect_timeout=1.0,
                probe_timeout=1.0,
                tenant_key_file=None,
            ),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(scene="single_instance")
            ),
            instance_id=0,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_enabled,
            block_size=4,
        ),
        model_config=SimpleNamespace(served_model_name="test", model="test"),
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req",
        client_index=0,
        skip_reading_prefix_cache=False,
        prompt_token_ids=list(range(16)),
        num_prompt_tokens=16,
        mm_features=[],
    )


class _Client:
    def __init__(
        self,
        target: LwdConnectionId,
        calls: list[tuple[int, int]],
        activity: dict[str, int],
    ) -> None:
        self.target = target
        self.calls = calls
        self.activity = activity

    def build_manifest(
        self, request_id: str, prompt_token_ids: list[int], media_items: tuple
    ) -> LwdPrefixManifest:
        return LwdPrefixManifest(
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            block_size=4,
            full_block_hashes=(bytes(32), bytes(32), bytes(32), bytes(32)),
            tail_hash=None,
        )

    async def probe(
        self, _session: object, request_id: str, _prompt: list[int], **kwargs
    ):
        self.calls.append((self.target.cloud_id, self.target.dp_idx))
        self.activity["active"] += 1
        self.activity["max_active"] = max(
            self.activity["max_active"], self.activity["active"]
        )
        await asyncio.sleep(0.02)
        self.activity["active"] -= 1
        hit_blocks = self.target.dp_idx + 1
        return LwdProbeResult(
            request_id=request_id,
            instance_id="cloud-0",
            block_size=4,
            hit_blocks=hit_blocks,
            hit_tokens=hit_blocks * 4,
            cloud_id=self.target.cloud_id,
            engine_index=self.target.dp_idx,
        )

    async def activate(
        self, _session: object, _request_id: str, _prompt: list[int], **kwargs
    ):
        assert kwargs["edge_id"] == 0
        assert kwargs["dp_idx"] == self.target.dp_idx


def _drain(manager: LwdEdgeProbeManager, activated: bool = False):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        completions = manager.drain()
        if completions:
            assert completions[0].activated is activated
            return completions[0]
        time.sleep(0.005)
    raise AssertionError("probe completion did not arrive")


@pytest.mark.parametrize(
    ("prefix_enabled", "expected_probes", "expected_concurrency"),
    [(True, 2, 2), (False, 1, 1)],
)
def test_submit_fans_out_only_for_prefix_and_activates_winner(
    prefix_enabled: bool, expected_probes: int, expected_concurrency: int
) -> None:
    manager = LwdEdgeProbeManager(_config(prefix_enabled), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    clients = {target: _Client(target, calls, activity) for target in manager._targets}
    manager._client = lambda target, _block_size: clients[target]
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert len(calls) == expected_probes
        assert len(completion.candidates) == expected_probes
        assert activity["max_active"] == expected_concurrency

        winner = max(
            completion.candidates,
            key=lambda candidate: candidate.normalized_hit_tokens(16),
        )
        assert manager.activate(completion, winner.connection_id, winner)
        activated = _drain(manager, activated=True)
        assert activated.connection_id == winner.connection_id
        assert activated.result == winner.result
        assert activated.probe_nonce == completion.probe_nonce
    finally:
        manager.close()


def test_submit_failure_cleans_inflight_and_falls_back() -> None:
    class _StoppedRuntime:
        def submit_group(self, _coroutines):
            raise RuntimeError("event loop stopped")

        def stop(self) -> None:
            pass

    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    manager._runtime = _StoppedRuntime()

    assert manager.submit(_request()) is LwdProbeSubmission.UNAVAILABLE
    assert not manager._inflight
    manager.close()


def test_resubmission_supersedes_old_batch_and_advances_nonce() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    clients = {target: _Client(target, calls, activity) for target in manager._targets}
    manager._client = lambda target, _block_size: clients[target]
    try:
        request = _request()
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.probe_nonce == 2
        assert len(manager._inflight) == 1
    finally:
        manager.close()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

from typing import Annotated, Any

from pydantic import ConfigDict, Field, StrictBool, model_validator
from pydantic.dataclasses import dataclass

from .topology import LwdString, _reject_legacy_coordination_fields


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordinationConfig:
    enabled: StrictBool = False
    placement_only: StrictBool = False
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    listen_host: LwdString = "0.0.0.0"
    instance_id: LwdString = "cloud-0"
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )

    probe_timeout: Annotated[float, Field(strict=True, gt=5.0, allow_inf_nan=False)] = (
        25.0
    )
    enforce_mm_abi_match: StrictBool = True

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, raw: Any) -> Any:
        return _reject_legacy_coordination_fields(raw)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdCoordinationConfig":
        _reject_legacy_coordination_fields(raw)
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if (
            self.enabled
            and role == "edge"
            and not self.tenant_key_file
            and not self.placement_only
        ):
            raise ValueError(
                "[LWD] edge coordination requires tenant_key_file "
                "unless placement_only is true"
            )

"""Tests for the weight-transfer protocol factory."""

import textwrap
from argparse import Namespace

import pytest

from miles.backends.training_utils.weight_update.protocol import (
    WeightTransferProtocol,
    get_weight_transfer_protocol,
)


def test_custom_mode_loads_protocol_from_path(tmp_path, monkeypatch):
    (tmp_path / "my_transfer_proto.py").write_text(
        textwrap.dedent(
            """
            from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol


            class MyProtocol(WeightTransferProtocol):
                def connect(
                    self,
                    rollout_engines,
                    rollout_engine_lock,
                    engine_gpu_counts,
                    engine_gpu_offsets,
                    parallel_state,
                    placement,
                ):
                    pass

                def send_bucket(self, bucket, weight_version):
                    pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args = Namespace(
        update_weight_transfer_mode="custom",
        custom_weight_transfer_path="my_transfer_proto.MyProtocol",
    )
    protocol = get_weight_transfer_protocol(args)
    assert isinstance(protocol, WeightTransferProtocol)
    assert protocol.args is args
    assert not protocol.is_fresh()


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="bogus"):
        get_weight_transfer_protocol(Namespace(update_weight_transfer_mode="bogus"))

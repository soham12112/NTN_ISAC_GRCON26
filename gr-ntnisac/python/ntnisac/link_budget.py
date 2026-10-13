#!/usr/bin/env python3
"""Message-only bistatic link-budget block."""

import pmt
from gnuradio import gr

from .dwell import add_repo_py, pack, unpack

add_repo_py()


class link_budget(gr.basic_block):
    """Publish an honest far-illuminator echo feasibility calculation.

    The ratio is sigma/(4*pi*Rr^2): satellite range and transmit power cancel
    against the direct path.  A configured result is published on start, and a
    ``query`` message can override any field at runtime.
    """

    def __init__(self, rcs_m2=1000.0, receiver_range_m=500.0,
                 direct_cn_db=15.0, bandwidth_hz=4.5e6, dwell_s=0.1,
                 threshold_db=13.0, implementation_loss_db=0.0):
        gr.basic_block.__init__(
            self, name="link_budget", in_sig=None, out_sig=None)
        self.defaults = dict(
            rcs_m2=float(rcs_m2),
            receiver_range_m=float(receiver_range_m),
            direct_cn_db=float(direct_cn_db),
            bandwidth_hz=float(bandwidth_hz),
            dwell_s=float(dwell_s),
            threshold_db=float(threshold_db),
            implementation_loss_db=float(implementation_loss_db),
        )
        self.in_port = pmt.intern("query")
        self.out_port = pmt.intern("result")
        self.message_port_register_in(self.in_port)
        self.message_port_register_out(self.out_port)
        self.set_msg_handler(self.in_port, self._query)

    def _publish(self, values):
        from ntn_isac import link_budget as budget

        cfg = dict(self.defaults)
        cfg.update({k: float(v) for k, v in values.items() if k in cfg})
        result = budget.assess(**cfg)
        result.update(cfg)
        self.message_port_pub(self.out_port, pack("link_budget", result))

    def _query(self, msg):
        _, values = unpack(msg)
        if isinstance(values, dict):
            self._publish(values)

    def start(self):
        self._publish({})
        return super().start()

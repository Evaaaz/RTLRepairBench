"""Offline fallback for the LLM lane.

`canned_rtl(prompt)` returns a small, self-contained, SYNTHESIZABLE
SystemVerilog module so the rest of the pipeline (RTL generation, lint,
report) runs end-to-end with no model server present.

This module is pure stdlib: importing it must never require torch / vllm /
openai / transformers. It is the last-resort path for backend.llm_client.
"""

from __future__ import annotations

# A parameterized INT8 multiply-accumulate (MAC) processing element.
# Synthesizable, no SystemVerilog constructs that Verilator rejects in
# --lint-only mode. Signed 8-bit operands, widened accumulator.
_CANNED_INT8_MAC = """// STUB - replace in backend.serve_stub / backend.llm_client (real model output)
// Parameterized signed INT8 multiply-accumulate processing element.
// Generated offline as a Verilator-lintable fallback.
module int8_mac #(
    parameter int unsigned DATA_W = 8,
    parameter int unsigned ACC_W  = 32
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       en,
    input  logic                       clear,
    input  logic signed [DATA_W-1:0]   a,
    input  logic signed [DATA_W-1:0]   b,
    output logic signed [ACC_W-1:0]    acc
);

    // Sign-extended product of the two INT8 operands.
    logic signed [2*DATA_W-1:0] product;
    assign product = a * b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= '0;
        end else if (clear) begin
            acc <= '0;
        end else if (en) begin
            acc <= acc + ACC_W'(product);
        end
    end

endmodule
"""


def canned_rtl(prompt: str) -> str:
    """Return a small synthesizable SystemVerilog module.

    The ``prompt`` is accepted for signature parity with the real LLM path
    but is not interpreted; this is a deterministic offline stub.
    """
    # STUB - replace in backend.serve_stub (real model output via llm_client)
    return _CANNED_INT8_MAC


module Prob013_m2014_q4e_ref (
  input in1,
  input in2,
  output logic out
);

  assign out = ~(in1 | in2);

endmodule


module Prob010_mt2015_q4a_ref (
  input x,
  input y,
  output z
);

  assign z = (x^y) & x;

endmodule

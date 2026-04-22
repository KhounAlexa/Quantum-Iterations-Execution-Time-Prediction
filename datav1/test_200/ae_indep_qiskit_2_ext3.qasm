OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
ry(2.34619382340565) q[1];
cry(2.34619382340565) q[0],q[1];
h q[0];

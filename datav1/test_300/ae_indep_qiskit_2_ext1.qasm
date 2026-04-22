OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
ry(0.7953988301841436) q[1];
cry(0.7953988301841436) q[0],q[1];
h q[0];

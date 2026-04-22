OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
ry(1.4706289056333368) q[1];
cry(1.4706289056333368) q[0],q[1];
h q[0];

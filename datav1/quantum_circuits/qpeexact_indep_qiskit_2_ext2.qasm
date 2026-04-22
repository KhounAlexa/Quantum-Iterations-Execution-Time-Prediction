OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
cp(4*pi/5) q[0],q[1];
h q[0];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
cp(2.8198869717398165) q[0],q[1];
h q[0];

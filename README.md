# Verification-IP-for-Costas-Loop-with-Machine-Learning-
Implemented a UVM testbench to interface it with Machine Learning Algortihms. The UVM testbench is still under progress. Meanwhile, I have created a Python Script to parse the existing VCD file generated with the System Verilog testbench for Costas Loop DUT. The script parses VCD file for Sin and cos values and groups it in pairs of 6 integers which creates one wave.

Each wave is then overlapped making 12 such waves resulting in an eye Diagram. These eye diagrams are classified as Good and Bad, and then a model is trained using SVC, which gives an accuracy of 95%.

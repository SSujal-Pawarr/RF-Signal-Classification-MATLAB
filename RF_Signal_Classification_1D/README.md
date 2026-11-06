# Final 1-D RF Signal Classification — MATLAB R2026b

Four-class simulated RF classification using a 1-D CNN:
WLAN, Bluetooth Low Energy, LoRa, FM.

INPUT
- 1024 I/Q samples per example
- column 1 = I (In-phase)
- column 2 = Q (Quadrature)
- the CNN receives a 2 x 1024 sequence

DEMO
Run:
    cd('/MATLAB Drive/RF_Signal_Classification_1D_Final')
    runProject

The menu supports:
1 WLAN test sample
2 BluetoothLE test sample
3 LoRa test sample
4 FM test sample
5 Random held-out test sample
6 Custom I/Q MAT file
7 Complete test-set evaluation
8 Exit

VISUAL OUTPUT
The demo shows raw I and Q, I-Q distribution, 1-D frequency spectrum, and classification scores. Figure labels/axes are set to black.

CUSTOM INPUTS
The custom_inputs folder contains four realistic-style synthetic noisy I/Q files plus custom_test_input.mat. They are NOT actual SDR/antenna recordings. A real-world claim requires measured RF data from suitable acquisition hardware or an independent real RF dataset.

ACCURACY
Test accuracy is measured on the held-out simulated test set. It is not real-world RF accuracy.

FILES
runProject.m
generateLabeledWirelessDataset.m
trainWirelessClassifier.m
testWirelessClassifier.m
demoWirelessIQ.m
howtorun.m
START_HERE.txt
custom_inputs/*.mat

REQUIREMENT
MATLAB R2026b or compatible release with Deep Learning Toolbox.

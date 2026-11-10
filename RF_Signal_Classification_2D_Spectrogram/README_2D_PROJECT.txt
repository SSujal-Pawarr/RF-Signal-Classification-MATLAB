AI-BASED RF SIGNAL CLASSIFICATION - 2D SPECTROGRAM VERSION
============================================================
SEPARATE from the previous 1-D raw-I/Q project.

Classes: WLAN, BluetoothLE, LoRa, FM
Input: simulated complex I/Q
2-D representation: STFT spectrogram (time x frequency)
Model: 2-D CNN
Hardware: none

RUN:
1) cd('/MATLAB Drive/RF_Signal_Classification_2D_Spectrogram')
2) runProject2D

After training:
demo2DRFSignal

Custom input:
Use a MAT file containing numeric variable iq, preferably 1024 x 2.
Column 1 = I, column 2 = Q.

The project uses simulated data. Test accuracy does not imply real-world RF performance.

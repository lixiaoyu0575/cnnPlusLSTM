# cnnPlusLSTM
Based on http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/ and lstm

#Note
1.use checkData.py to make sure that the source data from "http://www.cis.fordham.edu/wisdm/dataset.php"(the actitracker part, the WISDM_at_v2.0_raw.txt) is fine. You need to remove the ";" each line, delete the lines without xyz data, and also make sure the six type of activities are included. I have extract a small group of data named "actitracker_new9.txt" with more than 10 thousand lines to test the code.
2.the preprocess code is also separated into preprocessData.py to save the formalized data in "processData" document.
3.the py file in "CNN" and "CNNLSTM" is the main code, reading the preprocessed data from "processedData" document, organizing the net construction, recording running loss and accuracy which is recorded in tfVis/logs/nn_logs
4.run "tensorboard --logdir=path/to/nn_logs" and check the graph in http://127.0.1.1:6006

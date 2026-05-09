This code assumes we have the rgb-d image from a highly accurate fixed point aerial positioned camera. We would use this to capture the ground truth and compare the locally calculated volume change from the visual system on the excavator. 

This can be used because we are using a small-scale toy excavator, so the 860mm x 860mm field of it can be accurately captured by using a high ed rgb-d from an aerial view. So, for every time step we observe scoop movmeent in the excvator we collect the rgb-d image as well. 


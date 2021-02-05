import pandas as pd

def init_record():
    record = pd.DataFrame({'ot_loss_rec_frame':[],'ot_loss_raw_frame':[],
                           'ot_loss_rec_frame_w':[],'ot_loss_raw_frame_w':[],
                           'ot_loss_rec_frame_mid':[],'ot_loss_raw_frame_mid':[],
                           'ot_loss_rec_burst':[],'ot_loss_raw_burst':[],
                           'ot_loss_rec_burst_w':[],'ot_loss_raw_burst_w':[],
                           'psnr_ave':[],'psnr_std':[]})
    return record
    

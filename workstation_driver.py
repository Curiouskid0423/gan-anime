from main import Driver
import inference

if __name__ == '__main__':
    driver = Driver('/ckpts', '/logs')
    driver.train(prev_ckpt=None, n_epoch=100) 
    inference(workspace_dir='.', inference_ckpt_path='/ckpts/latest_ckpt.pth')
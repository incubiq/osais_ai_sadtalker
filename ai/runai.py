
##
##      SADTALKER AI
##

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, './ai')

## sadtalker specifics
import torch, uuid
import shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from pydub import AudioSegment

def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")

class SadTalker():

    def __init__(self, checkpoint_path='ai/checkpoints', config_path='ai/src/config', lazy_load=False):

        if torch.cuda.is_available() :
            device = "cuda"
        else:
            device = "cpu"
        
        self.device = device

        os.environ['TORCH_HOME']= checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path


    def initDir(self, indir, outdir):
        self.indir=indir
        self.outdir=outdir
        self.tempdir="./_temp"
      

    def test(self, output_image, source_image, driven_audio, preprocess='crop', 
        still_mode=False,  use_enhancer=False, batch_size=1, size=256, 
        pose_style = 0, exp_scale=1.0, 
        use_ref_video = False,
        ref_video = None,
        ref_info = None,
        use_idle_mode = False,
        length_of_audio = 0, use_blink=True,
        result_dir='./results/'):

        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, size, False, preprocess)
        print(self.sadtalker_paths)
            
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        # time_tag = str(uuid.uuid4())
        # save_dir = os.path.join(result_dir, time_tag)
        # os.makedirs(save_dir, exist_ok=True)

        # input_dir = os.path.join(save_dir, 'input')
        # os.makedirs(input_dir, exist_ok=True)

        # print(source_image)
        pic_path = os.path.join(self.indir, os.path.basename(source_image)) 
        # shutil.move(source_image, self.indir)

        if driven_audio is not None and os.path.isfile(os.path.join(self.indir, driven_audio)):
            audio_path = os.path.join(self.indir, os.path.basename(driven_audio))  

            #### mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            # else:
            #    shutil.move(driven_audio, self.indir)

        elif use_idle_mode:
            audio_path = os.path.join(self.indir, 'idlemode_'+str(length_of_audio)+'.wav') ## generate audio from this new audio_path
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  #duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all': # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(self.outdir, ref_video_videoname+'.wav')
            print('new audiopath:',audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
            os.system(cmd)        

        os.makedirs(self.outdir, exist_ok=True)
        
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(self.tempdir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(self.outdir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ =  self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':            
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        #audio2ceoff
        if use_ref_video and ref_info == 'all':
            coeff_path = ref_video_coeff_path # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink) # longer audio?
            coeff_path = self.audio_to_coeff.generate(batch, self.tempdir, pose_style, ref_pose_coeff_path)

        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess, size=size, expression_scale = exp_scale)
        data["video_name"]=output_image.replace('.mp4', '')
        return_path = self.animate_from_coeff.generate(data, self.tempdir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size)

        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        import gc; gc.collect()

        _fileRet=os.path.join(self.outdir, output_image)
        
        ## encode video for browser compat
        #try: 
        #    cmd = r"ffmpeg -i %s -vcodec h264 %s"%(return_path, _fileRet)
        #    os.system(cmd)        
        #except:

        try: 
            cmd = r"ffmpeg -y -i %s -ar 22050 -ab 512k -b 800k -f mp4 -s %i*%i -strict -2 -c:a aac %s"%(return_path, size, size, _fileRet)
            os.system(cmd)

            # as we convert format... we wait a bit (1sec) to ensure file is properly saved (or it risks sending a partial video)
            import time
            time.sleep(1)
        except:
            shutil.copy2(return_path, _fileRet)

        print(f'The generated video is: {_fileRet}')
        os.remove(return_path)
        return _fileRet

    

# init the global session
sad_talker = SadTalker('ai/checkpoints', 'ai/src/config', True)

## where to save the user profile?
def fnGetUserdataPath(_username):
    _path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_PROFILE_DIR = os.path.join(_path, '_profile')
    USER_PROFILE_DIR = os.path.join(DEFAULT_PROFILE_DIR, _username)
    return {
        "location": USER_PROFILE_DIR,
        "voice": False,
        "picture": True
    }

## WARMUP Data
def getWarmupData(_id):
    try:
        import time
        from werkzeug.datastructures import MultiDict
        ts=int(time.time())
        sample_args = MultiDict([
            ('-u', 'test_user'),
            ('-uid', str(ts)),
            ('-t', _id),
            ('-cycle', '0'),
            ('-o', 'warmup.mp4'),
            ('-filename', 'warmup.jpg'),
            ('-audio', 'warmup.wav')
        ])
        return sample_args
    except Exception as err:
        print("Could not call warm up!\r\n")
        return None

def fnRun(_args): 
    global sad_talker

    try:
        vq_parser = argparse.ArgumentParser()

        # OSAIS arguments
        vq_parser.add_argument("-odir", "--outdir", type=str, help="Output directory", default="./_output/", dest='outdir')
        vq_parser.add_argument("-idir", "--indir", type=str, help="input directory", default="./_input/", dest='indir')

        # Add the SADTalker arguments
        vq_parser.add_argument("-filename","--init_image", type=str, help="Initial image filename", default="warmup.jpg", dest='init_image')
        vq_parser.add_argument("-audio","--init_audio", type=str, help="Initial audio filename", default="warmup.wav", dest='init_audio')
        vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.mp4", dest='output')
        vq_parser.add_argument("-still", "--still_mode", type=str, help="Still Mode?", default="False", dest='is_still_mode')
        vq_parser.add_argument("-enhance", "--use_enhancer", type=str, help="Use Enhancer?", default="False", dest='use_enhancer')
        vq_parser.add_argument("-blink", "--use_blink", type=str, help="Use Blink?", default="True", dest='use_blink')
        vq_parser.add_argument("-res", "--res", type=int, help="resolution", default=256, dest='resolution')
        vq_parser.add_argument("-cimg", "--batch_size", type=int, help="How many output", default=1, dest='batch_size')
        vq_parser.add_argument("-pose", "--pose_style", type=int, help="Pose style", default=1, dest='pose_style')
        vq_parser.add_argument("-type", "--preprocess_type", type=str, help="Preprocessing type", default="crop", dest='preprocess_type')

        args = vq_parser.parse_args(_args)
        print(args)

        beg_date = datetime.utcnow()

        _use_enhancer = (args.use_enhancer=="True" or args.use_enhancer=="true")
        _is_still_mode = (args.is_still_mode=="True" or args.is_still_mode=="true")
        _use_blink = (args.use_blink=="True" or args.use_blink=="true")

        sad_talker.initDir(args.indir, args.outdir)
        _resFile=sad_talker.test(args.output, args.init_image, args.init_audio, args.preprocess_type, _is_still_mode, _use_enhancer, args.batch_size, args.resolution, args.pose_style)
        sys.stdout.flush()

        ## return output
        end_date = datetime.utcnow()
        return {
            "beg_date": beg_date,
            "end_date": end_date,
            "aFile": [_resFile]
        }
    
    except Exception as err:
        print("\r\nCRITICAL ERROR!!!")
        raise err


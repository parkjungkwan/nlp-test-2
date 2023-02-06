import numpy as np
import torch
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
import dlib
import imageio
from scipy.spatial import ConvexHull
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import yaml
from tqdm import tqdm

from model.encoder.align_all_parallel import align_face
from model.encoder.psp import pSp
from model.dualstylegan import DualStyleGAN
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork


class Dsg:
    def __init__(self):
        self.name = 'winter'
        self.device = torch.device('cuda:0')
        self.model_dir = './checkpoints'
        self.data_dir = './data'
        self.output_dir = './generated'
        self.style_type_dir = f'{self.model_dir}/cartoon'
        self.image_path = f'{self.data_dir}/{self.name}.jpg'
        self.toonify_image_path = f'{self.data_dir}/{self.name}_toonify.png'
        self.driving_video_path = f'{self.data_dir}/driving.mp4'
        self.output_video_path = f'{self.output_dir}/{self.name}.mp4'
        self.config_path = f'{self.model_dir}/vox-256.yaml'
        self.checkpoint_path = f'{self.model_dir}/vox.pth.tar'
        self.style_id = 26
        self.w = [0.6]*7+[1]*11
        self.pixel = 256

    def visualize(self, img_arr):
        plt.imshow(((img_arr.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.axis('off')

    
    def load_model(self):
        generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        generator.eval()
        ckpt = torch.load(f'{self.style_type_dir}/generator.pt', map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        generator = generator.to(self.device)
        return generator
        
    def load_encoder(self):
        model_path = f'{self.model_dir}/encoder.pt'
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        opts.device = self.device
        encoder = pSp(opts)
        encoder.eval()
        encoder = encoder.to(self.device)
        return encoder
        
    def alignment(self):
        modelname = f'{self.model_dir}/shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(modelname)
        aligned_image = align_face(filepath=self.image_path, predictor=predictor)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        alignment_image = transform(aligned_image).unsqueeze(dim=0).to(self.device)
        plt.figure(figsize=(10, 10), dpi=30)
        self.visualize(alignment_image[0].cpu())
        plt.savefig(f'./data/{self.name}_align.png')
        return alignment_image
    
    def image_grid_generate(self):
        encoder = self.load_encoder()
        alignment_image = self.alignment()
        generator = self.load_model()
        exstyles = np.load(f'{self.style_type_dir}/refined_exstyle_code.npy', allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[self.style_id]
        with torch.no_grad():
            img_rec, instyle = encoder(alignment_image, randomize_noise=False, return_latents=True,
                                z_plus_latent=True, return_z_plus_latent=True, resize=False)
            latent = torch.tensor(exstyles[stylename]).repeat(2, 1, 1).to(self.device)
            latent[1, 7:18] = instyle[0, 7:18]
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
        
        results = []
        for i in range(6):
            for j in range(6):
                w = [i/5.0]*7+[j/5.0]*11
                img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                        truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)
                results += [img_gen]
                
        vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 6, 1)
        plt.figure(figsize=(10, 10), dpi=120)
        self.visualize(vis.cpu())
        plt.savefig(f'{self.data_dir}/{self.name}_grid.png', bbox_inches='tight')

    def image_one_generate(self):
        encoder = self.load_encoder()
        alignment_image = self.alignment()
        generator = self.load_model()
        exstyles = np.load(f'{self.style_type_dir}/refined_exstyle_code.npy', allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[self.style_id]
        visualize = Dsg().visualize
        img_rec, instyle = encoder(alignment_image, randomize_noise=False, return_latents=True,
                                   z_plus_latent=True, return_z_plus_latent=True, resize=False)
        latent = torch.tensor(exstyles[stylename]).repeat(2, 1, 1).to(self.device)
        latent[1, 7:18] = instyle[0, 7:18]
        exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
            latent.shape)
        img_gen, _ = generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                               truncation=0.7, truncation_latent=0, use_res=True, interp_weights=self.w)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)

        visualize(img_gen[0].cpu())
        plt.savefig(f'{self.data_dir}/{self.name}_toonify.png', bbox_inches='tight')

    def load_source(self):
        warnings.filterwarnings("ignore")
        source_image = imageio.imread(self.toonify_image_path)
        reader = imageio.get_reader(self.driving_video_path)
        self.toonify_image_path = resize(source_image, (self.pixel, self.pixel))[..., :3]

        self.fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        self.driving_video = [resize(frame, (self.pixel, self.pixel))[..., :3] for frame in driving_video]

    def load_checkpoints(self):
        device = self.device
        with open(self.config_path) as f:
            config = yaml.full_load(f)

        inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                       **config['model_params']['common_params'])
        kp_detector = KPDetector(**config['model_params']['common_params'])
        dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                                  **config['model_params']['dense_motion_params'])
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                                 **config['model_params']['avd_network_params'])
        kp_detector.to(device)
        dense_motion_network.to(device)
        inpainting.to(device)
        avd_network.to(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        inpainting.load_state_dict(checkpoint['inpainting_network'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        if 'avd_network' in checkpoint:
            avd_network.load_state_dict(checkpoint['avd_network'])

        inpainting.eval()
        kp_detector.eval()
        dense_motion_network.eval()
        avd_network.eval()

        return inpainting, kp_detector, dense_motion_network, avd_network

    def relative_kp(self, kp_source, kp_driving, kp_driving_initial):

        source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        kp_new = {k: v for k, v in kp_driving.items()}

        kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
        kp_value_diff *= adapt_movement_scale
        kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

        return kp_new

    def make_animation(self, source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network,
                       device, mode='relative'):
        assert mode in ['standard', 'relative', 'avd']
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            source = source.to(device)
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(
                device)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                driving_frame = driving_frame.to(device)
                kp_driving = kp_detector(driving_frame)
                if mode == 'standard':
                    kp_norm = kp_driving
                elif mode == 'relative':
                    kp_norm = self.relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                          kp_driving_initial=kp_driving_initial)
                elif mode == 'avd':
                    kp_norm = avd_network(kp_source, kp_driving)
                dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param=None,
                                                    dropout_flag=False)
                out = inpainting_network(source, dense_motion)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    def make_video(self):
        warnings.filterwarnings("ignore")
        source_image = imageio.imread(self.toonify_image_path)
        reader = imageio.get_reader(self.driving_video_path)
        source_image = resize(source_image, (self.pixel, self.pixel))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        driving_video = [resize(frame, (self.pixel, self.pixel))[..., :3] for frame in driving_video]

        inpainting, kp_detector, dense_motion_network, avd_network = self.load_checkpoints()
        device = self.device
        predictions = self.make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                     avd_network, device=device, mode='relative')

        imageio.mimsave(self.output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    # Dsg().alignment()
    # Dsg().image_grid_generate()
    # Dsg().image_one_generate()
    Dsg().make_video()


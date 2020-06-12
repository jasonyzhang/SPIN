import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import neural_renderer as nr
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid


colors = {
    # colorblind/print/copy safe:
    'blue': [0.65098039, 0.74117647, 0.85882353],
    'pink': [.9, .7, .7],
    'mint': [ 166 / 255.,  229 / 255.,  204 / 255.],
    'mint2': [ 202 / 255.,  229 / 255.,  223 / 255.],
    'green': [ 153 / 255.,  216 / 255.,  201 / 255.],
    'green2': [ 171 / 255.,  221 / 255.,  164 / 255.],
    'red': [ 251 / 255.,  128 / 255.,  114 / 255.],
    'orange': [ 253 / 255.,  174 / 255.,  97 / 255.],
    'yellow': [ 250 / 255.,  230 / 255.,  154 / 255.]
}


def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X
    proj = X

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)

class VisRenderer(object):
    """
    Utility to render meshes using pytorch NMR
    faces are F x 3 or 1 x F x 3 numpy
    this is for visualization only -- does not allow backprop.
    This class assumes all inputs are Torch/numpy variables.
    This renderer expects quarternion rotation for camera,,
    """

    def __init__(self,
                 img_res=224,
                 t_size=1):

        self.renderer = nr.Renderer(
            img_res, camera_mode='look_at', perspective=False)
        self.set_light_dir([1, .5, -1], int_dir=0.3, int_amb=0.7)
        self.set_bgcolor([1, 1, 1.])
        self.img_size = img_res

        self.faces_np = np.load('/home/jasonzh2/hmr_pytorch/models/smpl_faces.npy').astype(np.int)
        self.faces = asVariable(torch.IntTensor(self.faces_np).cuda())
        # self.faces = asVariable(torch.IntTensor(faces.astype(int)).cuda())

        if self.faces.dim() == 2:
            self.faces = torch.unsqueeze(self.faces, 0)

        # Default color:
        default_tex = np.ones((1, self.faces.shape[1], t_size, t_size, t_size,
                               3))
        self.default_tex = asVariable(torch.FloatTensor(default_tex).cuda())

        # Default camera:
        cam = np.hstack([0.9, 0, 0])
        default_cam = asVariable(torch.FloatTensor(cam).cuda())
        self.default_cam = torch.unsqueeze(default_cam, 0)

        # Setup proj fn:
        self.proj_fn = orthographic_proj_withz_idrot

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(
                self.__call__(vertices[i], camera_translation[i], images_np[i]),
                (2, 0, 1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self,
                 verts,
                 cam=None,
                 texture=None,
                 rend_mask=False,
                 alpha=False,
                 img=None,
                 color_name='blue'):
        """
        verts is |V| x 3 numpy/cuda torch Variable
        cams is 3D [s, tx, ty], numpy/cuda torch Variable
        cams is NOT the same as OpenDR renderer.
        Directly use the cams of HMR output
        Returns N x N x 3 numpy
        """
        if texture is None:
            # single color.
            color = torch.FloatTensor(colors[color_name]).cuda()
            texture = color * self.default_tex
        else:
            texture = asFloatTensor(texture)
            if texture.dim() == 5:
                # Here input it F x T x T x T x 3 (instead of F x T x T x 3)
                # So add batch dim.
                texture = torch.unsqueeze(texture, 0)
        if cam is None:
            cam = self.default_cam
        else:
            cam = asFloatTensor(cam)
            if cam.dim() == 1:
                cam = torch.unsqueeze(cam, 0)

        verts = asFloatTensor(verts)
        if verts.dim() == 2:
            verts = torch.unsqueeze(verts, 0)

        verts = asVariable(verts)
        cam = asVariable(cam)
        texture = asVariable(texture)

        # set offset_z for persp proj
        proj_verts = self.proj_fn(verts, cam, offset_z=0)
        # Flipping the y-axis here to make it align with
        # the image coordinate system!
        proj_verts[:, :, 1] *= -1
        if rend_mask:
            rend = self.renderer.render_silhouettes(proj_verts, self.faces)
            rend = rend.repeat(3, 1, 1)
            rend = rend.unsqueeze(0)
        else:
            rend = self.renderer.render(proj_verts, self.faces, texture)

        rend = rend[0].data.cpu().numpy()[0].transpose((1, 2, 0))
        rend = np.clip(rend, 0, 1) * 255.0

        if not rend_mask and (alpha or img is not None):
            mask = self.renderer.render_silhouettes(proj_verts, self.faces)
            mask = mask[0].data.cpu().numpy()
            if img is not None:
                mask = np.repeat(np.expand_dims(mask, 2), 3, axis=2)
                # TODO: Make sure img is [0, 255]!!!
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint)
                return (img * (1 - mask) + rend * mask).astype(np.uint8)
            else:
                return self.make_alpha(rend, mask)
        else:
            return rend.astype(np.uint8)

    def rotated(self,
                verts,
                deg,
                axis='y',
                cam=None,
                texture=None,
                rend_mask=False,
                alpha=False,
                color_name='blue'):
        """
        vert is N x 3, torch FloatTensor (or Variable)
        """
        import cv2
        if axis == 'y':
            axis = [0, 1., 0]
        elif axis == 'x':
            axis = [1., 0, 0]
        else:
            axis = [0, 0, 1.]

        new_rot = cv2.Rodrigues(np.deg2rad(deg) * np.array(axis))[0]
        new_rot = asFloatTensor(new_rot)

        verts = asFloatTensor(verts)
        center = verts.mean(0)
        new_verts = torch.t(torch.matmul(new_rot,
                                         torch.t(verts - center))) + center

        return self.__call__(
            new_verts,
            cam=cam,
            texture=texture,
            rend_mask=rend_mask,
            alpha=alpha,
            color_name=color_name
        )

    def make_alpha(self, rend, mask):
        rend = rend.astype(np.uint8)
        alpha = (mask * 255).astype(np.uint8)

        imgA = np.dstack((rend, alpha))
        return imgA

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):
        self.renderer.light_direction = direction
        self.renderer.light_intensity_directional = int_dir
        self.renderer.light_intensity_ambient = int_amb

    def set_bgcolor(self, color):
        self.renderer.background_color = color


def asVariable(x):
    if type(x) is not torch.autograd.Variable:
        x = Variable(x, requires_grad=False)
    return x


def asFloatTensor(x):
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x).cuda()
    # ow assumed it's already a Tensor..
    return x


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    if type(trg) is torch.autograd.Variable:
        src = Variable(src, requires_grad=False)
    return src


def make_square(img):
    """
    Bc nmr only deals with square image, adds pad to the shorter side.
    """
    img_size = np.max(img.shape[:2])
    pad_vals = img_size - img.shape[:2]

    img = np.pad(
        array=img,
        pad_width=((0, pad_vals[0]), (0, pad_vals[1]), (0, 0)),
        mode='constant'
    )

    return img, pad_vals


def remove_pads(img, pad_vals):
    """
    Undos padding done by make_square.
    """

    if pad_vals[0] != 0:
        img = img[:-pad_vals[0], :]
    if pad_vals[1] != 0:
        img = img[:, :-pad_vals[1]]

    return img


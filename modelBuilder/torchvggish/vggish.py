import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vggish_input, vggish_params
import math



class GAB(nn.Module):
    def __init__(self):
        super(GAB, self).__init__()
        self.GAB_layer= nn.Sequential(
            nn.AvgPool2d((6,4)),
            nn.Conv2d(512,512,1),
            nn.ReLU(True),
            nn.Conv2d(512,512,1),
            nn.Sigmoid(),
        )
    def forward(self,inputs:torch.tensor):
        x=self.GAB_layer(inputs)
        C_A=torch.mul(x,inputs)
        x=torch.mean(C_A,dim=0,keepdims=True)
        x=torch.nn.functional.sigmoid(x)
        S_A=torch.mul(C_A, x)
        return S_A

class CAB(nn.Module):
    def __init__(self, k, classes):
        super(CAB, self).__init__()
        self.CAB_layer=nn.Sequential(
            nn.Conv2d(512, k*classes, 1),
            nn.BatchNorm2d(k*classes),
            nn.ReLU(),
        )
        self.k=k
        self.classes=classes

    
    def forward(self,inputs:torch.tensor):
        # print("aaa")
        F=self.CAB_layer(inputs)
        
        x=nn.functional.max_pool2d(F,  kernel_size=(inputs.shape[2],inputs.shape[3]))
        x=torch.reshape(x, (x.shape[0],self.classes,self.k,1,1))
        S=torch.mean(x,dim=-3,keepdim=False)


        x=torch.reshape(F, (F.shape[0],self.k,self.classes,F.shape[2],F.shape[3]))
        x=torch.mean(x,dim=1,keepdim=False)

        x=torch.mul(S,x)

        M=torch.mean(x,dim=1,keepdim=True)
        semantic=torch.mul(M,inputs)

        return semantic




class VGG(nn.Module):
    def __init__(self, features, use_attention, k,classes, use_resize=True):
        super(VGG, self).__init__()
        kernel_size_1= 1 if vggish_params.NUM_FRAMES < vggish_params.STANDARD_NUM_FRAMES else vggish_params.NUM_FRAMES % vggish_params.STANDARD_NUM_FRAMES +1;
        kernel_size_2= 1 if vggish_params.NUM_BANDS < vggish_params.STANDARD_NUM_BANDS else vggish_params.NUM_BANDS % vggish_params.STANDARD_NUM_BANDS +1;
        stride_1 = max(1, int(vggish_params.NUM_FRAMES/vggish_params.STANDARD_NUM_FRAMES))
        stride_2 = max(1, int(vggish_params.NUM_BANDS / vggish_params.STANDARD_NUM_BANDS))
        # todo 目前只实现了 偶数的小图片的padding
        padding_1 = max(0, int((vggish_params.STANDARD_NUM_FRAMES - vggish_params.NUM_FRAMES)/2))
        padding_2 = max(0, int((vggish_params.STANDARD_NUM_BANDS - vggish_params.NUM_BANDS)/2))
        # 利用卷积 回到 96*64
        self.resize_conv = nn.Conv2d(1, 1, kernel_size=(kernel_size_1 ,
                            kernel_size_2), 
                            stride=(stride_1,
                             stride_2), 
                            padding=(padding_1, padding_2))

        self.features = features
        
        # self.GAB_layer= nn.Sequential(
        #     nn.AvgPool2d((6,4)),
        #     nn.Conv2d(512,512,1),
        #     nn.ReLU(True),
        #     nn.Conv2d(512,512,1),
        #     nn.Sigmoid(),
        # )

        # self.CAB_layer=nn.Sequential(
        #     nn.Conv2d(512, k*classes, 1),
        #     nn.BatchNorm2d(k*classes),
        #     nn.ReLU(),
        # )
        self.GAB_part= GAB()
        self.CAB_part= CAB(k, classes)

        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

        self.use_attention=use_attention
        self.use_resize=use_resize



    # def GAB(self,inputs:torch.tensor):
    #     x=self.GAB_layer(inputs)
    #     C_A=torch.mul(x,inputs)
        
        
    #     x=torch.mean(C_A,dim=0,keepdims=True)
    #     x=torch.nn.functional.sigmoid(x)
    #     S_A=torch.mul(C_A, x)
    #     return S_A

    # def CAB(self,inputs:torch.tensor):
    #     F=self.CAB_layer(inputs)
        
    #     x=nn.functional.max_pool2d(F,  kernel_size=(inputs.shape[2],inputs.shape[3]))
    #     x=torch.reshape(x, (x.shape[0],self.classes,self.k,1,1))
    #     S=torch.mean(x,dim=-3,keepdim=False)


    #     x=torch.reshape(F, (F.shape[0],self.k,self.classes,F.shape[2],F.shape[3]))
    #     x=torch.mean(x,dim=1,keepdim=False)

    #     x=torch.mul(S,x)

    #     # print(x.shape)
    #     # print(inputs.shape)
    #     M=torch.mean(x,dim=1,keepdim=True)
    #     semantic=torch.mul(M,inputs)

    #     return semantic
        
    def forward(self, x):
        # print('aaa')
        # print(x.shape)
        #feature 提取出来之后就是 1*96*64
        if self.use_resize:
            x=self.resize_conv(x)
        # print(x.shape)
        #feature 提取出来之后就是 channel*6*4
        x = self.features(x) 
        # print(x.shape)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        if self.use_attention:
            # print('GAB&CAB')
            x=self.GAB_part(x)
            # print(x.shape)
            x=self.CAB_part(x)
        
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


# def _spectrogram():
#     config = dict(
#         sr=16000,
#         n_fft=400,
#         n_mels=64,
#         hop_length=160,
#         window="hann",
#         center=False,
#         pad_mode="reflect",
#         htk=True,
#         fmin=125,
#         fmax=7500,
#         output_format='Magnitude',
#         #             device=device,
#     )
#     return Spectrogram.MelSpectrogram(**config)


# 多加了一个全连接层
class VGGish(VGG):
    def __init__(self, model_path, use_attention=False, k=5, num_class=2,device=None, pretrained=False, preprocess=False, postprocess=False, progress=True, use_resize=True):
        # print(use_resize)
        super().__init__(make_layers(),use_attention, k, num_class, use_resize=use_resize)
        if pretrained:
            state_dict = torch.load(model_path)
            super().load_state_dict(state_dict)

        # if device is None:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        # 加入全连接层
        self.fc=nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.to(self.device)

    def forward(self, x, fs=None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        # print(x.shape) #[num_examples, num_frames, num_bands]
        x = x.to(self.device)
        x = VGG.forward(self, x)
        # print(x.shape)
        x= self.fc(x)
        # print(x.shape)

        return x

    def _preprocess(self, x, fs):
        if isinstance(x, np.ndarray):
            x = vggish_input.waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = vggish_input.wavfile_to_examples(x)
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        return self.pproc(x)

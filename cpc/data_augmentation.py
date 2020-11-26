from typing import Tuple
import random
import torch
import torchaudio
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
import augment.sox_effects as sox_effects


class BandrejectAugment:
    def __init__(self, scaler=1.0):
        self.target_info = {
            "channels": 1,
            "length": 0,
            "precision": 16,
            "rate": 16000.0,
            "bits_per_sample": 16,
        }

        def random_band():
            low, high = BandrejectAugment.generate_freq_mask(scaler)
            return f"{high}-{low}"

        self.effect = (
            sox_effects.EffectChain()
            .denormalize()
            .sinc("-a", "120", random_band)
            .dither()
            .normalize()
        )

    @staticmethod
    def freq2mel(f):
        return 2595.0 * np.log10(1 + f / 700)

    @staticmethod
    def mel2freq(m):
        return (10.0 ** (m / 2595.0) - 1) * 700

    @staticmethod
    def generate_freq_mask(scaler):
        sample_rate = 16000.0  # TODO: configurable
        F = 27.0 * scaler
        melfmax = BandrejectAugment.freq2mel(sample_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.0)
        melf0 = np.random.uniform(0, melfmax - meldf)
        low = BandrejectAugment.mel2freq(melf0)
        high = BandrejectAugment.mel2freq(melf0 + meldf)

        return low, high

    def __call__(self, x):
        src_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        y, _ = self.effect.apply(x, src_info=src_info, target_info=self.target_info)

        return y


class PitchAugment:
    def __init__(self, quick=False, shift_max=300):
        """
        shift_max {int} -- shift in 1/100 of semi-tone (default: {100})
        """
        random_shift = lambda: np.random.randint(-shift_max, shift_max)
        effect = sox_effects.EffectChain().denormalize().pitch(random_shift)

        if quick:
            effect = effect.rate("-q", 16000)
        else:
            effect = effect.rate(16000)
        effect = effect.dither().normalize()
        self.effect = effect

    def __call__(self, x):
        target_info = {
            "channels": 1,
            # it might happen that the output has 1 frame more
            # by asking for the specific length, we avoid this
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        src_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        y, _ = self.effect.apply(x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        y = y.view_as(x)
        return y


class PitchDropout:
    def __init__(self, T_ms=100, shift_max=300):
        random_shift = lambda: np.random.randint(-shift_max, shift_max)
        effect = (
            sox_effects.EffectChain()
            .denormalize()
            .pitch(random_shift)
            .rate("-q", 16000)
            .dither()
        )
        effect = effect.normalize().time_dropout(max_seconds=T_ms / 1000.0)
        self.effect = effect

    def __call__(self, x):
        target_info = {
            "channels": 1,
            # it might happen that the output has 1 frame more
            # by asking for the specific length, we avoid this
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        src_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        y, _ = self.effect.apply(x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        y = y.view_as(x)
        return y


class ReverbAugment:
    def __init__(self, shift_max=100):
        random_room_size = lambda: np.random.randint(0, shift_max)
        self.effect = (
            sox_effects.EffectChain()
            .denormalize()
            .reverb("50", "50", random_room_size)
            .channels()
            .dither()
            .normalize()
        )

    def __call__(self, x):
        src_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        target_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 16,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        y, sr = self.effect.apply(x, src_info=src_info, target_info=target_info)

        y = y.view_as(x)
        return y


class AdditiveNoiseAugment:
    def __init__(self, noise_dataset, snr):
        assert noise_dataset and snr >= 0.0
        self.noise_dataset = noise_dataset
        r = np.exp(snr * np.log(10) / 10)
        self.t = r / (1.0 + r)
        self.update_noise_loader()

    def update_noise_loader(self):
        self.noise_data_loader = iter(
            self.noise_dataset.getDataLoader(1, "uniform", True)
        )

    def __call__(self, x):
        # idx = np.random.randint(0, len(self.noise_dataset))
        try:
            noise = next(self.noise_data_loader)[0]
        except StopIteration:
            self.update_noise_loader()
            noise = next(self.noise_data_loader)[0]
        # noise = self.noise_dataset[idx][0]
        # noise is non-augmented, we get two identical samples
        noise = noise[0, 0, ...]

        noised = self.t * x + (1.0 - self.t) * noise.view_as(x)
        return noised


class RandomAdditiveNoiseAugment:
    def __init__(self, snr=15):
        self.snr = np.exp(snr * np.log(10) / 10)

    def __call__(self, x):

        alpha = self.snr / x.std()
        noise = torch.randn(x.size(), device=x.device) / alpha
        return x + noise


class ReverbDropout:
    def __init__(self, T_ms=100):
        random_room_size = lambda: np.random.randint(0, 100)
        self.effect = (
            sox_effects.EffectChain()
            .denormalize()
            .reverb("50", "50", random_room_size)
            .channels()
            .dither()
            .normalize()
            .time_dropout(max_seconds=T_ms / 1000.0)
        )

    def __call__(self, x):
        src_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        target_info = {
            "channels": 1,
            "length": x.size(1),
            "precision": 16,
            "rate": 16000.0,
            "bits_per_sample": 32,
        }

        y, sr = self.effect.apply(x, src_info=src_info, target_info=target_info)

        y = y.view_as(x)
        return y


class TimeDropoutAugment:
    def __init__(self, T_ms=100, sr=16000):
        self.effect = sox_effects.EffectChain().time_dropout(max_seconds=T_ms / 1000.0)

    def __call__(self, x):
        y, sr = self.effect.apply(
            x, src_info={"rate": 16000.0}, target_info={"rate": 16000.0}
        )
        return y


class IDAugment:
    def __call__(self, x):
        return x


class AugmentCfg:
    def __init__(self, **kwargs):
        self.augment_type = kwargs["type"]
        self.config = {k: i for k, i in kwargs.items() if k != "type"}

    def __repr__(self):
        return f"{self.augment_type} : \n {self.config}"


class CombinedTransforms:
    def __init__(self, augment_cfgs=None, augment_transforms=None):

        assert bool(augment_cfgs is not None) ^ (augment_transforms is not None)

        if augment_cfgs is not None:
            self.transforms = [get_augment(x.augment_type, **x.config) for x in augment_cfgs]
        elif augment_transforms is not None:
            self.transforms = augment_transforms

    def __call__(self, x):
        transform = random.choice(self.transforms)
        return transform(x)


def get_augment(augment_type, **kwargs):
    if not augment_type or augment_type == "none":
        return IDAugment()
    elif augment_type == "bandreject":
        return BandrejectAugment(**kwargs)
    elif augment_type == "pitch":
        return PitchAugment(**kwargs)
    elif augment_type == "reverb":
        return ReverbAugment(**kwargs)
    elif augment_type == "time_dropout":
        return TimeDropoutAugment(**kwargs)
    elif augment_type == "reverb_dropout":
        return ReverbDropout(**kwargs)
    elif augment_type == "random_noise":
        return RandomAdditiveNoiseAugment(**kwargs)
    elif augment_type in ["pitch_dropout"]:
        return PitchDropout(**kwargs)
    else:
        raise RuntimeError(f"Unknown augment_type = {augment_type}")


def augmentation_factory(args, noise_dataset=None):
    if not args.augment_type or args.augment_type == ["none"]:
        return None

    transforms = []
    for augment_type in args.augment_type:
        if augment_type == "none":
            transforms.append(IDAugment())
        elif augment_type == "bandreject":
            transforms.append(BandrejectAugment(scaler=args.bandreject_scaler))
        elif augment_type in ["pitch", "pitch_quick"]:
            transforms.append(PitchAugment(quick=args.augment_type == "pitch_quick"))
        elif augment_type == "reverb":
            transforms.append(ReverbAugment())
        elif augment_type == "time_dropout":
            transforms.append(TimeDropoutAugment(args.t_ms))
        elif augment_type == "additive":
            if not noise_dataset:
                raise RuntimeError("Noise dataset is needed for the additive noise")
            transforms.append(
                AdditiveNoiseAugment(noise_dataset, args.additive_noise_snr)
            )
        elif augment_type == "reverb_dropout":
            transforms.append(ReverbDropout(args.t_ms))
        elif augment_type in ["pitch_dropout"]:
            transforms.append(PitchDropout(args.t_ms))
        else:
            transforms.append(
                RuntimeError(f"Unknown augment_type = {args.augment_type}")
            )

    if len(transforms) == 1:
        return transforms[0]

    return CombinedTransforms(augment_transforms=transforms)

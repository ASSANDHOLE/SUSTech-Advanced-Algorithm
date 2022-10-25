import os
import random
import shutil


def create_animated_gif(plt_img_list: list, save_path: str, duration: int = 100, loop: int = 0) -> None:
    """
    Parameters
    ----------
    plt_img_list : list
        A list of plots by matplotlib.
    save_path : str
        The path to save the gif.
    duration : int
        The duration of each frame. The unit is ms.
    loop : int
        The number of loops. 0 means infinite.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('Pillow is not installed. Please install it by running "pip install Pillow"')
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    while os.path.exists(temp_path):
        temp_path += str(random.randint(0, 9))
    os.mkdir(temp_path)
    for i, plt_img in enumerate(plt_img_list):
        plt_img.savefig(os.path.join(temp_path, f'img{i}.png'))
    images = []
    for i in range(len(plt_img_list)):
        images.append(Image.open(os.path.join(temp_path, f'img{i}.png')))
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=duration, loop=loop)
    shutil.rmtree(temp_path)

import os

from PIL import Image, ImageFont, ImageDraw


def create_binary_overlay(image_to_annotate, prediction, root):
    output_directory = root + "Images_for_annotation/"
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    font = ImageFont.truetype("./signlib/Arcon.otf", 25)
    img = Image.open(root + str(image_to_annotate))
    draw = ImageDraw.Draw(img)
    if prediction == 0:
        overlay = "1 Handed"
        position = (20, 20)
    else:
        overlay = "2 Handed"
        position = (20, 50)
    draw.text(position, str(overlay), (255, 255, 0), font=font)
    draw = ImageDraw.Draw(img)
    img.save(output_directory + str(image_to_annotate) + ".png")
    # os.remove('./Outputs/openpos/result_'+str(j)+'.png')

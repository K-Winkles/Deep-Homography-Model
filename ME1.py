# Machine Exercise # 1

def superimpose(image_path, **kwargs):
    from PIL import Image
    from PIL import ImageDraw
    img = Image.open(image_path).convert('RGBA')
    img_copy = img.copy()
    coords = list()

    for key, value in kwargs.items():
        coords.append(value)

    draw = ImageDraw.Draw(img_copy)
    draw.polygon(coords, fill='blue')

    final_image = Image.blend(img, img_copy, 0.5)
    final_image.save('superpoly.png')



superimpose(
    'bulbatest.png',
    a = (500,5000),
    b = (-400,1000),
    c = (-750, 500),
    d = (100,-1000),
)
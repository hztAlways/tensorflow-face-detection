# 导入必要的库
import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf

# 设置标题的左边距和上边距
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

# 定义标准颜色列表，用于绘制边框和文本背景
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def save_image_array_as_png(image, output_path):
  """将图像（表示为 numpy 数组）保存为 PNG 格式。

  Args:
    image: 一个形状为 [height, width, 3] 的 numpy 数组。
    output_path: 要保存图像的路径。
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')

def encode_image_array_as_png_str(image):
  """将 numpy 数组编码为 PNG 字符串。

  Args:
    image: 一个形状为 [height, width, 3] 的 numpy 数组。

  Returns:
    PNG 编码的图像字符串。
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string

def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """在图像（numpy 数组）上添加边框框。

  Args:
    image: 一个形状为 [height, width, 3] 的 numpy 数组。
    ymin: 边框框的最小 y 坐标（归一化坐标）。
    xmin: 边框框的最小 x 坐标。
    ymax: 边框框的最大 y 坐标。
    xmax: 边框框的最大 x 坐标。
    color: 绘制边框框的颜色，默认为红色。
    thickness: 线条粗细，默认值为 4。
    display_str_list: 显示在框中的字符串列表（每个字符串显示在单独的一行上）。
    use_normalized_coordinates: 如果为 True（默认值），则将坐标视为相对于图像的归一化坐标，否则视为绝对坐标。
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """在图像上添加边框框。

  每个 display_str_list 中的字符串显示在边框框上方的单独一行中，以黑色文字显示在填充颜色的矩形内。

  Args:
    image: 一个 PIL.Image 对象。
    ymin: 边框框的最小 y 坐标。
    xmin: 边框框的最小 x 坐标。
    ymax: 边框框的最大 y 坐标。
    xmax: 边框框的最大 x 坐标。
    color: 绘制边框框的颜色，默认为红色。
    thickness: 线条粗细，默认值为 4。
    display_str_list: 显示在框中的字符串列表（每个字符串显示在单独的一行上）。
    use_normalized_coordinates: 如果为 True（默认值），则将坐标视为相对于图像的归一化坐标，否则视为绝对坐标。
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  text_bottom = top
  # 反转列表并从下到上打印。
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """在图像（numpy 数组）上绘制边框框。

  Args:
    image: 一个 numpy 数组对象。
    boxes: 一个形状为 [N, 4] 的二维 numpy 数组：（ymin, xmin, ymax, xmax）。坐标为归一化格式，范围为 [0, 1]。
    color: 绘制边框框的颜色，默认为红色。
    thickness: 线条粗细，默认值为 4。
    display_str_list_list: 字符串列表的列表。每个边框框的字符串列表。传递字符串列表的原因是它可能包含多个标签。

  Raises:
    ValueError: 如果 boxes 不是 [N, 4] 数组
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """在图像上绘制边框框。

  Args:
    image: 一个 PIL.Image 对象。
    boxes: 一个形状为 [N, 4] 的二维 numpy 数组：（ymin, xmin, ymax, xmax）。坐标为归一化格式，范围为 [0, 1]。
    color: 绘制边框框的颜色，默认为红色。
    thickness: 线条粗细，默认值为 4。
    display_str_list_list: 字符串列表的列表。每个边框框的字符串列表。

  Raises:
    ValueError: 如果 boxes 不是 [N, 4] 数组
  """
  if not boxes.shape:
    return
  if len(boxes.shape) != 2 or boxes.shape[1] != 4:
    raise ValueError('`boxes` 必须是一个形状为 [N, 4] 的二维 numpy 数组。')
  for i in range(boxes.shape[0]):
    display_str_list = () if display_str_list_list is None else display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i][0], boxes[i][1], boxes[i][2],
                               boxes[i][3], color, thickness, display_str_list)

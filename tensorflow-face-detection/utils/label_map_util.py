# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Label map utility functions."""

import logging
# 导入日志记录模块

import tensorflow as tf
# 导入 TensorFlow 模块
from google.protobuf import text_format
# 导入 Protocol Buffers 的 text_format 模块，用于解析文本格式的协议缓冲区
from protos import string_int_label_map_pb2
# 导入 string_int_label_map_pb2 模块，包含定义的协议缓冲区消息类型

# 这段代码是一个用于验证标签映射的有效性的函数。它接受一个 StringIntLabelMap 对象作为输入，并检查其中的每个标签条目的 id 是否有效。如果任何一个标签的 id 小于 1，函数将抛出 ValueError 异常，提示标签映射的 id 应该大于等于 1。
def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 1:
      raise ValueError('Label map ids should be >= 1.')
    # 遍历标签映射中的每个条目，如果 id 小于 1，则抛出 ValueError

# 这段代码定义了一个函数 create_category_index，用于创建一个 COCO 兼容的类别字典，以类别 id 为键。
def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  # 创建一个空字典 category_index
  for cat in categories:
    category_index[cat['id']] = cat
    # 遍历输入的类别列表，将每个类别的 'id' 作为键，类别字典作为值，添加到 category_index 中
  return category_index
  # 返回构建的类别字典

# 这个函数的作用是将标签映射（label map）转换为与评估兼容的类别列表。在目标检测任务中，通常需要将标签映射转换为一组类别，以便评估检测结果。该函数返回一个列表，其中每个元素都是一个字典，包含以下键值对：
# - `'id'`: 必需，表示类别的整数 id，用于唯一标识类别。
# - `'name'`: 必需，表示类别的名称，通常是字符串，例如'cat'、'dog'、'pizza'等。
# 该函数的参数如下：
# - `label_map`: StringIntLabelMapProto 或 None 类型。如果为 None，则创建一个具有最大类别数量的默认类别列表。
# - `max_num_classes`: 最大的类别数量，用于限制返回的类别列表。
# - `use_display_name`: 布尔值，选择是否使用标签映射中的 `display_name` 字段作为类别名称。如果为 False 或者 `display_name` 字段不存在，则使用 `name` 字段作为类别名称。
# 函数首先检查是否存在标签映射，如果没有则创建一个具有默认类别的列表。然后，它遍历标签映射中的每个条目，将满足条件的条目添加到类别列表中。如果 `use_display_name` 为 True，并且条目包含 `display_name` 字段，则使用 `display_name` 作为类别名称；否则使用 `name` 字段。在添加类别时，还会检查是否已经添加过相同 id 的类别，如果是，则只添加第一个。最终返回包含所有可能类别的列表。
def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
  """Loads label map proto and returns categories list compatible with eval.

  This function loads a label map and returns a list of dicts, each of which
  has the following keys:
    'id': (required) an integer id uniquely identifying this category.
    'name': (required) string representing category name
      e.g., 'cat', 'dog', 'pizza'.
  We only allow class into the list if its id-label_id_offset is
  between 0 (inclusive) and max_num_classes (exclusive).
  If there are several items mapping to the same id in the label map,
  we will only keep the first one in the categories list.

  Args:
    label_map: a StringIntLabelMapProto or None.  If None, a default categories
      list is created with max_num_classes categories.
    max_num_classes: maximum number of (consecutive) label indices to include.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.
  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  categories = []
  # 创建一个空列表 categories，用于存储转换后的类别
  list_of_ids_already_added = []
  # 创建一个空列表 list_of_ids_already_added，用于记录已添加的类别 id
  if not label_map:
    # 如果 label_map 为空，则创建默认类别列表
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
    # 返回创建的默认类别列表
  for item in label_map.item:
    # 遍历标签映射中的每个条目
    if not 0 < item.id <= max_num_classes:
      logging.info('Ignore item %d since it falls outside of requested '
                   'label range.', item.id)
      continue
    # 如果条目的 id 不在有效范围内，则忽略该条目
    if use_display_name and item.HasField('display_name'):
      name = item.display_name
    else:
      name = item.name
    # 根据 use_display_name 参数选择使用 display_name 或 name 作为类别名称
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      categories.append({'id': item.id, 'name': name})
    # 如果该 id 尚未添加过，则添加到类别列表中
  return categories
  # 返回转换后的类别列表

# 这个函数用于加载标签映射（label map）协议缓冲区，并返回一个 StringIntLabelMapProto 对象。它的参数是 path，指定了标签映射的路径。
# 函数的实现如下：
# 使用 TensorFlow 的 tf.io.gfile.GFile 打开指定路径的文件，以读取其中的内容。
# 将读取的内容解析为字符串，并存储在 label_map_string 中。
# 创建一个空的 StringIntLabelMap 对象 label_map。
# 尝试使用 text_format.Merge 将 label_map_string 中的内容合并到 label_map 中。如果解析失败，会捕获 text_format.ParseError 异常。
# 如果合并失败（即解析失败），则使用 ParseFromString 方法将 label_map_string 中的内容解析为 label_map 对象。
# 最后，调用 _validate_label_map 函数验证加载的标签映射是否有效，并返回 label_map 对象。
def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map
  # 返回加载的标签映射对象

# 这个函数用于读取标签映射文件并返回一个字典，该字典将标签名称映射到标签 ID。它的参数是 `label_map_path`，指定了标签映射文件的路径。
# 函数的实现如下：
# 1. 使用 `load_labelmap` 函数加载指定路径的标签映射文件，得到一个 `label_map` 对象。
# 2. 创建一个空字典 `label_map_dict`，用于存储标签名称到标签 ID 的映射关系。
# 3. 遍历 `label_map` 中的每一个条目，对于每一个条目，将其名称作为字典的键，ID 作为字典的值，添加到 `label_map_dict` 中。
# 4. 返回 `label_map_dict` 字典。
# 这个函数的作用是方便地读取标签映射文件，并将其中的标签名称和对应的 ID 存储在一个字典中，以便后续使用。
def get_label_map_dict(label_map_path):
  """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path: path to label_map.

  Returns:
    A dictionary mapping label names to id.
  """
  label_map = load_labelmap(label_map_path)
  # 加载指定路径的标签映射文件，得到 label_map 对象
  label_map_dict = {}
  # 创建一个空字典 label_map_dict
  for item in label_map.item:
    label_map_dict[item.name] = item.id
    # 遍历标签映射中的每个条目，将其名称作为键，ID 作为值，添加到字典中
  return label_map_dict
  # 返回构建的标签映射字典

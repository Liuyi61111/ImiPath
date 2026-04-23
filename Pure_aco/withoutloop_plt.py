import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

def delete_loops(visited_nodes_path):
    ''' Checks if there is a loop in the resulting path and deletes it '''
    path_withloop = list(visited_nodes_path)
    for element in path_withloop:
        # coincidence 是列表; 获取element重复的位置
        coincidences = get_coincidence_indices(path_withloop, element)
        # reverse de list to delete elements from back to front of the list
        coincidences.reverse()  # 用于反向列表中元素
        for i,coincidence in enumerate(coincidences):
            if not i == len(coincidences)-1:  # 判断到最后一个元素都已经操作过了
                path_withloop[coincidences[i+1]:coincidence] = []  # 将coincidence[i+1]到coincidence[i]之间的路径删除
    return path_withloop

def get_coincidence_indices(path_withloop, element):
    ''' Gets the indices of the coincidences of elements in the path '''
    result = []
    offset = -1
    while True:
        try:
            offset = path_withloop.index(element, offset+1)  # 可能产生异常的代码块
            # 在offset中从 offset+1 位置开始寻找重复的element的位置
        except ValueError:
            return result
        result.append(offset)


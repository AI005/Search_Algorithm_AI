import pygame
import graphUI
from node_color import white, yellow, black, red, blue, purple, orange, green, grey
import random
from collections import deque
from queue import PriorityQueue
import math
import sys
"""
Feel free print graph, edges to console to get more understand input.
Do not change input parameters
Create new function/file if necessary
"""
time_delay = 100

def fill_edge(v_from, v_to, edges, edge_id, color):
    edges[edge_id(v_from, v_to)][1] = color

def fill_vertex(graph, vertex, color):
    graph[vertex][3] = color


def fill_Path(path, edges, edge_id, color):
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = color


def BFS(graph, edges, edge_id, start, goal):

    n = len(graph)  # number of vertex
    visited = [False for _ in range(n)]
    prev = [-1 for _ in range(n)]
    queue = deque()

    queue.append(start)
    
    visited[start] = True

    # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)
    
    graphUI.updateUI()
    pygame.time.delay(time_delay)
    
    # --- Solve ---
    while True:
        graphUI.updateUI()

        if len(queue) == 0:
            break

        current = queue.popleft()

        #fill color explored edge to white
        if current != start:
            fill_edge(current, prev[current], edges, edge_id, white)
        
        if current == goal:
            fill_vertex(graph, current, blue)
            graphUI.updateUI()
            break
            
        

        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)

        neighbors = graph[current][1]
        for neighbor in neighbors:
            if not(visited[neighbor]):
                queue.append(neighbor)
                visited[neighbor] = True
                prev[neighbor] = current

                # set color node will be visited and edge among us
                fill_vertex(graph, neighbor, red)
                fill_edge(current, neighbor, edges, edge_id, green)

        graphUI.updateUI()
        pygame.time.delay(time_delay)
        
        # set color node was visted
        fill_vertex(graph, current, blue)

    # --- reconstructor path ---
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = prev[at]


    # Fill color path
    fill_Path(path, edges, edge_id, orange)
    graphUI.updateUI()


def redorder_list(l): 
    return random.sample(l, len(l))


def DFS(graph, edges, edge_id, start, goal):
    n = len(graph)  # number of vertex
    path = [start]
    stack = deque()
    
    stack.append(start)
    visited = [False for _ in range(n)]

    visited[start] = True
    is_found = False

    # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)  
    graphUI.updateUI()
    pygame.time.delay(time_delay)

    while len(stack) != 0:
        current = stack[-1]


        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)

        #fill color explored edge to white
        if len(path) >= 1 and path[-1] != current:
            fill_edge(current, path[-1], edges, edge_id, white)
       

        if not(visited[current]):
            path.append(current)
            visited[current] = True

            if current == goal:
                fill_vertex(graph, current, blue)
                graphUI.updateUI()
                break

        flag = False
        for next in redorder_list(graph[current][1]):
            if not(visited[next]):
                stack.append(next)

                # set color node will be visited and edge among us
                fill_vertex(graph, next, red)
                fill_edge(current, next, edges, edge_id, green)
                flag = True
                break

        if not(flag):   
            last = path.pop()
            # set color to default if dead end
            fill_edge(path[-1], last, edges, edge_id, grey)
            fill_vertex(graph, current, black)

            stack.pop()

        pygame.time.delay(time_delay)
        graphUI.updateUI()
        
        # set color node was visted
        if flag:
            fill_vertex(graph, current, blue)

    # Fill color path
    fill_Path(path, edges, edge_id, orange)
    graphUI.updateUI()

def getTableCost(graph):
    result = []
    for a in graph:
        for nei in a[1]:
            print(graph.index(a), nei, cost(graph, graph.index(a), nei))
    
    
def UCS(graph, edges, edge_id, start, goal):
    q = PriorityQueue()
    explored = set()
    q.put((0, [start]))
    result = []
    getTableCost(graph)
    # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)  
    graphUI.updateUI()
    pygame.time.delay(time_delay)


    while not q.empty():
        node = q.get()
        current = node[1][-1]
        _cost = node[0]
        
        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)

        fill_Path(node[1], edges, edge_id, white)
        print(explored)
        print(q_temp)
        if current == goal:
            fill_vertex(graph, current, blue)
            result = node[1]
            break

        explored.add(current)
        neighbors = graph[current][1]

        for neighbor in neighbors:
            if neighbor not in explored:
                q.put((_cost + cost(graph, current, neighbor), node[1] + [neighbor]))
                # set color node will be visited and edge among us
                fill_vertex(graph, neighbor, red)
                fill_edge(current, neighbor, edges, edge_id, green)
        q_temp.sort()
        graphUI.updateUI()
        pygame.time.delay(time_delay)
        #set color visted vertex to blue
        fill_vertex(graph, current, blue)

    print(result)
    fill_Path(result, edges, edge_id, orange)
    graphUI.updateUI()


def heuristic(graph, v_from, v_to):
    x1, y1 = graph[v_from][0]
    x2, y2 = graph[v_to][0]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def cost(graph, v_from, v_to):
    try:
        if v_to not in graph[v_from][1]:
            raise Exception('NotLinkedVertexs')

        return heuristic(graph, v_from, v_to)

    except Exception as error:
        print(error)

def AStar(graph, edges, edge_id, start, goal):
    openset = set()
    closedset = set()
    current = start
    openset.add(current)
    n = len(graph)
    G = [0 for _ in range(n)]
    parent = [-1 for _ in range(n)]

    # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)  
    graphUI.updateUI()
    pygame.time.delay(time_delay)

    while len(openset) != 0:
        current = min(openset, key=lambda a: G[a] + heuristic(graph, goal, a))
        
        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)
        
        if current != start:
            fill_edge(current, parent[current], edges, edge_id, white)

        if current == goal:
            fill_vertex(graph, current, blue)
            graphUI.updateUI()
            break
        
        openset.remove(current)
        closedset.add(current)

        neighbors = graph[current][1]
        for neighbor in neighbors:
            if neighbor in closedset:
                continue
            
            # set color node will be visited and edge among us
            fill_vertex(graph, neighbor, red)
            fill_edge(current, neighbor, edges, edge_id, green)

            if neighbor in openset:
                new_g = G[current] + cost(graph, current, neighbor)
                if new_g < G[neighbor]:
                    G[neighbor] = new_g
                    parent[neighbor] = current
            else:
                G[neighbor] += cost(graph, current, neighbor)
                parent[neighbor] = current
                openset.add(neighbor)
        pygame.time.delay(time_delay)
        graphUI.updateUI()

        for v in closedset:
            fill_vertex(graph, v, blue)
 
 
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = parent[at]


    # Fill color path
    fill_Path(path, edges, edge_id, orange)
    graphUI.updateUI()
        

# Best first search
def BeFS(graph, edges, edge_id, start, goal):
    openset = set()
    closedset = set()
    current = start
    openset.add(current)
    parent = [-1 for _ in range(len(graph))]

    # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)  
    graphUI.updateUI()
    pygame.time.delay(time_delay)

    while len(openset) != 0:
        current = min(openset, key = lambda a: heuristic(graph, a, goal))

        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)

        if current != start:
            fill_edge(current, parent[current], edges, edge_id, white)
        
        if current == goal:
            fill_vertex(graph, current, blue)
            graphUI.updateUI()
            break

        openset.remove(current)
        closedset.add(current)
        
        neighbors = graph[current][1]
        for neighbor in neighbors:
            if neighbor in closedset:
                continue

            # set color node will be visited and edge among us
            fill_vertex(graph, neighbor, red)
            fill_edge(current, neighbor, edges, edge_id, green)
            openset.add(neighbor)
            parent[neighbor] = current

        pygame.time.delay(time_delay)
        graphUI.updateUI()

        for v in closedset:
            fill_vertex(graph, v, blue)   

    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = parent[at]


    # Fill color path
    fill_Path(path, edges, edge_id, orange)
    graphUI.updateUI()


def example_func(graph, edges, edge_id, start, goal):
    """
    This function is just show some basic feature that you can use your project.
    @param graph: list - contain information of graph (same value as global_graph)
                    list of object:
                     [0] : (x,y) coordinate in UI
                     [1] : adjacent node indexes
                     [2] : node edge color
                     [3] : node fill color
                Ex: graph = [
                                [
                                    (139, 140),             # position of node when draw on UI
                                    [1, 2],                 # list of adjacent node
                                    (100, 100, 100),        # grey - node edged color
                                    (0, 0, 0)               # black - node fill color
                                ],
                                [(312, 224), [0, 4, 2, 3], (100, 100, 100), (0, 0, 0)],
                                ...
                            ]
                It means this graph has Node 0 links to Node 1 and Node 2.
                Node 1 links to Node 0,2,3 and 4.
    @param edges: dict - dictionary of edge_id: [(n1,n2), color]. Ex: edges[edge_id(0,1)] = [(0,1), (0,0,0)] : set color
                    of edge from Node 0 to Node 1 is black.
    @param edge_id: id of each edge between two nodes. Ex: edge_id(0, 1) : id edge of two Node 0 and Node 1
    @param start: int - start vertices/node
    @param goal: int - vertices/node to search
    @return:
    

    Ex1: Set all edge from Node 1 to Adjacency node of Node 1 is green edges.
    node_1 = graph[1]
    for adjacency_node in node_1[1]:
        edges[edge_id(1, adjacency_node)][1] = green
    graphUI.updateUI()

    Ex2: Set color of Node 2 is Red
    graph[2][3] = red
    graphUI.updateUI()

    Ex3: Set all edge between node in a array.
    path = [4, 7, 9]  # -> set edge from 4-7, 7-9 is blue
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = blue
    graphUI.updateUI()
    pass"""
    pass

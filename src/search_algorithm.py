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
time_delay = 700

def fill_edge(v_from, v_to, edges, edge_id, color):
    edges[edge_id(v_from, v_to)][1] = color

def fill_vertex(graph, vertex, color):
    graph[vertex][3] = color


def fill_Path(path, edges, edge_id, color):
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = color

def notify_no_find_path():
    try:
        raise Exception("No find path")
    except Exception as error:
        print(error)

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

        graphUI.updateUI()
        pygame.time.delay(time_delay)
        
        # set color node was visted
        fill_vertex(graph, current, blue)

    # --- reconstructor path ---
    if prev[goal] == -1:
        notify_no_find_path()
        
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = prev[at]


    # Fill color path
    fill_Path(path, edges, edge_id, green)
    graphUI.updateUI()

# Create a new list which random index from a list
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
        if len(path) >= 1 and current != path[-1]:
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
                fill_edge(current, next, edges, edge_id, orange)
                flag = True
                break

        if not(flag):   
            last = path.pop()
            # set color to default if dead end
            if len(path) >= 1:
                fill_edge(path[-1], last, edges, edge_id, grey)
            fill_vertex(graph, current, black)

            stack.pop()

        pygame.time.delay(time_delay)
        graphUI.updateUI()
        
        # set color node was visted
        if flag:
            fill_vertex(graph, current, blue)


    if len(path) == 0:
        notify_no_find_path()
    
    # Fill color path
    fill_Path(path, edges, edge_id, green)
    graphUI.updateUI()

    
def UCS(graph, edges, edge_id, start, goal):
    
    frontier = [start]
    parent = [-1 for _ in range(len(graph))]
    explored  = set()
    G = [-1 for _ in range(len(graph))]
    G[start] = 0
    
     # fill color started vertex to orange
    fill_vertex(graph, start, orange)
    # fill color goal vertex to purple
    fill_vertex(graph, goal, purple)  
    graphUI.updateUI()
    pygame.time.delay(time_delay)
    
    while len(frontier) != 0:
        
        # set current  = vertex x from frontier which G[x] is min
        current = min(frontier, key=lambda a: G[a])
        _cost = G[current]
        
        # fill color current vertex to yellow
        fill_vertex(graph, current, yellow)
        #fill color explored edge to white
        if parent[current] != -1: 
            fill_edge(current, parent[current], edges, edge_id, white)
            
        if current == goal:
            fill_vertex(graph, current, blue)
            graphUI.updateUI()
            break
        
        frontier.remove(current)
        explored.add(current)
        neighbors = graph[current][1]
        
        for neighbor in neighbors:
            if neighbor not in explored:
                if neighbor not in frontier:
                    fill_edge(current, neighbor,edges, edge_id, orange)
                    fill_vertex(graph, neighbor, red)
                    frontier.append(neighbor)
                    G[neighbor] = _cost + cost(graph,current, neighbor)
                    parent[neighbor] = current
                else:
                    if G[neighbor] > G[current] + cost(graph,current, neighbor):
                        G[neighbor] = _cost + cost(graph,current, neighbor)
                        parent[neighbor] = current
        
        
        pygame.time.delay(time_delay)
        graphUI.updateUI()
        fill_vertex(graph, current, blue)
    
    if parent[goal] == -1:
        notify_no_find_path()   
    
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = parent[at]
    
    fill_Path(path, edges, edge_id, green)
    graphUI.updateUI()
    print(path)
    
# heuristic calculate distance of two vertexs
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
            fill_edge(current, neighbor, edges, edge_id, orange)

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

    if parent[goal] == -1:
        notify_no_find_path()
        
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = parent[at]


    # Fill color path
    fill_Path(path, edges, edge_id, green)
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
            fill_edge(current, neighbor, edges, edge_id, orange)
            openset.add(neighbor)
            parent[neighbor] = current

        pygame.time.delay(time_delay)
        graphUI.updateUI()

        for v in closedset:
            fill_vertex(graph, v, blue)   
            
    if parent[goal] == -1:
        notify_no_find_path()
    
    # find path    
    path = []
    at = goal
    while at != -1:
        path.append(at)
        at = parent[at]


    # Fill color path
    fill_Path(path, edges, edge_id, green)
    graphUI.updateUI()


def example_func(graph, edges, edge_id, start, goal):       
    pass

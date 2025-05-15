#P1: A* Search Algorithm
import heapq

def A_star_search(start, goal, heuristic, get_neighbors, distance):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    return None  # No path found

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

    
#Sample Input and Output
start_node = (0, 0)
goal_node = (5, 5)
heuristic = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
get_neighbors = lambda node: [(node[0]+1, node[1]), (node[0]-1, node[1]), (node[0], node[1]+1), (node[0], node[1]-1)]
distance = lambda x, y: 1  # Assuming uniform cost for simplicity
result = A_star_search(start_node, goal_node, heuristic, get_neighbors, distance)
print("Path:", result)

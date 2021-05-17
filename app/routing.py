from driving import Route


class RouteFinder:

    class PathWrapper:
        def __init__(self, path):
            self.path = path
            self.prev = None

    @staticmethod
    def find_route(start, end):
        routes = RouteFinder._find_routes(start, end)
        shortest = RouteFinder._find_shortest_route(routes)
        shortest_way = RouteFinder._create_shortest_way(shortest)
        shortest_way = [start, *shortest_way]
        return Route(shortest_way)

    @staticmethod
    def _find_routes(start, end):
        start = RouteFinder.PathWrapper(start)
        end = end

        open_list = [start]
        closed_list = []
        routes = []

        while len(open_list) > 0:
            x = open_list.pop(0)
            if x.path.handle == end.handle:
                routes.append(x)
                continue
            closed_list.append(x)
            for neighbor in x.path.successors:
                if RouteFinder._check_if_closed(neighbor, closed_list):
                    continue
                else:
                    t = RouteFinder.PathWrapper(neighbor)
                    t.prev = x
                    open_list.append(t)
        return routes

    @staticmethod
    def _check_if_closed(p, closed):
        for item in closed:
            if item.path.handle == p.handle:
                return True
        return False

    @staticmethod
    def _calc_route_length(route):
        length = 0
        r = route.prev
        while True:
            if r is None:
                return length
            length += r.path.length
            r = r.prev

    @staticmethod
    def _find_shortest_route(routes):
        min_length = RouteFinder._calc_route_length(routes[0])
        min_route = routes[0]
        for route in routes:
            length = RouteFinder._calc_route_length(route)
            if length < min_length:
                min_length = length
                min_route = route
        return min_route

    @staticmethod
    def _create_shortest_way(shortest):
        arr = []
        r = shortest
        while r.prev is not None:
            arr.append(r.path)
            r = r.prev
        arr.reverse()
        return arr
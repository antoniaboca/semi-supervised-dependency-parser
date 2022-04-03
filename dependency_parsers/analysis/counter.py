import argparse
from dependency_parsers.data.processor import SentenceDataset
from dependency_parsers.data.params import ROOT_TOKEN, ROOT_TAG, ROOT_LABEL

class TagCounter():
    def __init__(self, file=None, sentence_set=None):
        if file is None and sentence_set is None:
            raise Exception('Provide either a file to load from or a sentence set')
        
        if file is not None and sentence_set is not None:
            raise Exception('Provide exactly one argument to work with')

        self.file = file
        self.sentence_set = sentence_set
        if file is not None:
            self.loaded = False
        else:
            self.loaded = True

        self.tree = {}
        self.graph = {}
        
    def load(self, sentence_length=None):
        print('Load the sentences from {}...'.format(self.file))
        self.sentence_set = SentenceDataset(self.file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL)
        self.vocabulary = self.sentence_set.getVocabulary()
        if sentence_length is not None:
            self.sentence_set = list(filter(lambda x: len(x[0]), self.sentence_set))
        self.sentence_set = self.sentence_set.sentences
        print('Loaded the sentence set. Number of sentences: {}'.format(len(self.sentence_set)))
        self.loaded = True

    def edge_count(self, tag_type):
        if self.loaded is False:
            raise Exception("Sentences not loaded yet.")

        if tag_type == 'xpos':
            t = 2
        elif tag_type == 'upos':
            t = 1
        else:
            raise Exception("I do not know the tag type")

        tree_edges = {}
        graph_edges = {}

        for sentence in self.sentence_set:
            for idx1 in range(len(sentence[0])):
                for idx2 in range(len(sentence[0])):
                    if idx1 == idx2:
                        continue
                    
                    xpos1 = sentence[t][idx1]
                    xpos2, parent2 = sentence[t][idx2], sentence[3][idx2]
                    
                    if (xpos1, xpos2) not in graph_edges:
                        graph_edges[(xpos1, xpos2)] = 0
                    graph_edges[(xpos1, xpos2)] += 1
                        
                    if parent2 == idx1:
                        if (xpos1, xpos2) not in tree_edges:
                            tree_edges[(xpos1, xpos2)] = 0
                        tree_edges[(xpos1, xpos2)] += 1

        self.tree, self.graph = tree_edges, graph_edges
        return tree_edges, graph_edges

    def top_edges(tree, graph, number, min_occurence):
        distribution = {}
        order = {}
        for edge in graph.keys():
            if graph[edge] < min_occurence:
                continue
            distribution[edge] = tree.get(edge, 0) / graph[edge]
        top = []
        for key, value in distribution.items():
            top.append((value, key))
        top.sort(reverse=True)
        distribution = {}

        idx = 0
        for value, key in top[:number]:
            distribution[key] = value
            order[key] = idx
            idx += 1
            
        return distribution, order
    
    def compare_to_prior(self, prior):
        graph_count = 0
        graph_total = 0
        tree_count = 0
        tree_total = 0
        for key, value in self.graph.items():
            if key in prior:
                graph_count += value
            graph_total += value
        
        for key, value in self.tree.items():
            if key in prior:
                tree_count += value
            tree_total += value
        
        return ((graph_count, graph_total), (tree_count, tree_total))
    
    def run_comparison(self, prior, tag_type):
        _, _ = self.edge_count(tag_type)
        (g_count, g_total), (t_count, t_total) = self.compare_to_prior(prior)
        print('\nPercetange of tree edges in prior: {:3.3f} ({}/{})'.format(t_count/t_total, t_count, t_total))
        print('\nPercetange of graph edges in prior:{:3.3f} ({}/{})'.format(g_count/g_total, g_count, g_total))
        return t_count/t_total, g_count/g_total

    def run_comparison_from_file(file, prior, tag_type, sentence_length=None):
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.run_comparison(prior, tag_type)
    
    def get_top_edges(self, number, min_occurence, tag_type):
        tree, graph = self.edge_count(tag_type)
        distribution, order = TagCounter.top_edges(tree, graph, number, min_occurence)
        
        for key, value in distribution.items():
            print('{}: {}'.format(key, value))
        return distribution, order
    
    def get_top_edges_from_file(file, number, min_occurence, tag_type, sentence_length=None):
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.get_top_edges(number, min_occurence, tag_type)
    
    def tag_search(self, f, min_occurence, tag_type):
        left = 1
        right = len(self.sentence_set)
        ans = 0
        while left < right:
            middle = (left + right) // 2
            dist, _ = self.get_top_edges(middle, min_occurence, tag_type)
            tree_f, _ = self.run_comparison(dist, tag_type)
            if tree_f < f:
                left = middle + 1
            else:
                ans = middle
                right = middle - 1
        
        return ans

    def tag_search_from_file(file, f, min_occurence, tag_type, sentence_length=None):
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.tag_search(f, min_occurence, tag_type)  
    
    def get_statistics_from_file(file, min_occurence, tag_type, sentence_length=None, step=5):
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.get_statistics(min_occurence, tag_type, step)

    def get_statistics(self, min_occurence, tag_type, step=5):
        d = {}
        for percentage in range(10,105,5):
            d[str(percentage) + '%'] = self.tag_search(percentage/100, min_occurence, tag_type)
        return d
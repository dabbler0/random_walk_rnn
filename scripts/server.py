#!/usr/bin/env python

import sys
sys.path.append('..')

import http.server
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch

import helpers
from models import predict

decoder = None
extractor = None
distributions = None

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)

        if self.path[:6] == '/image':
            self.send_header('Content-Type', 'image/png')
            self.end_headers()

            qs = parse_qs(self.path.split('?')[1])
            states, chars = (int(qs['states'][0]),
                    int(qs['chars'][0]))

            image_file = '/raid/lingo/abau/random-walks/dataset-%d-%d-0/graph-image.png' % (states, chars)

            with open(image_file,'rb') as f:
                self.wfile.write(f.read())

        elif self.path[:len('/load')] == '/load':
            global decoder, extractor

            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            qs = parse_qs(self.path.split('?')[1])

            states, chars, length = (int(qs['states'][0]),
                    int(qs['chars'][0]),
                    int(qs['length'][0]),
                    )

            decoder_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/epoch-0.pt' % (
                    states, chars, length
                    )
            extractor_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/extractor-model.pt' % (
                    states, chars, length
                    )
            gaussians_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/gaussians.json' % (
                    states, chars, length
                    )

            print('Loading %s, %s' % (decoder_filename, extractor_filename))

            decoder = torch.load(decoder_filename)
            extractor = torch.load(extractor_filename)

            ''' # TODO incorporate gaussians somehow
            if os.path.exists(gaussians_filename):
                with open(gaussians_filename) as f:
                    distributions = DistributionsRecord(json.load(f), graph)
            '''
            distributions = None

            self.wfile.write(json.dumps({'success': True}).encode('utf-8'))

            return

        elif self.path[:len('/interactive')] == '/interactive':
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            qs = parse_qs(self.path.split('?')[1])
            sentence = qs['sentence'][0]

            sentence = torch.LongTensor(
                [helpers.decoding_dict[c] for c in sentence]
            ).unsqueeze(0).cuda()

            inp = sentence[:, :-1]
            target = sentence[:, 1:]

            predictions, extractions, gaussians = predict.predict(decoder, extractor, distributions, inp, target)

            self.wfile.write(json.dumps({
                'predictions': [pred[0].cpu().numpy().tolist() for pred in predictions],
                'extractions': [ext[0].cpu().numpy().tolist() for ext in extractions]
                # TODO incorporate Gaussians somehow
            }).encode('utf-8'))

            return

        elif '?' in self.path:
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            qs = parse_qs(self.path.split('?')[1])
            states, chars, length, tlength = (int(qs['states'][0]),
                    int(qs['chars'][0]),
                    int(qs['length'][0]),
                    int(qs['tlength'][0]))

            annotation_file = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/sample-annotated.json' % (states, chars, length)
            samples_file = '/raid/lingo/abau/random-walks/testset-%d-%d-0/small_sample' % (states, chars)

            with open(samples_file) as f:
                sentences = f.read().split('\n')

            with open(annotation_file) as f:
                annotations = json.load(f)

            sentences = [(s, i) for i, s in enumerate(sentences) if len(s) == tlength]

            annotations = {j: annotations[str(i)] for j, (s, i) in enumerate(sentences)}

            sentences = [c[0] for c in sentences]

            self.wfile.write(json.dumps({
                'sentences': sentences,
                'annotations': annotations
            }).encode('utf-8'))

        else:
            self.send_header('Content-Type', 'text/html')
            self.end_headers()

            with open('visualize-running.html') as f:
                self.wfile.write(f.read().encode('utf-8'))

        return

server = HTTPServer(('127.0.0.1', 8080), Handler)

print('Serving')
server.serve_forever()

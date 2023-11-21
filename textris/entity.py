from . import question as q
import numpy as np
from copy import copy


class Entity:
    ''' An UMLS named entity class that makes the type or
    concept unique identifiers (TUIs or CUIs) directly accessible.
    '''
    
    def __init__(self, ent, linker=None):
        
        self.entity = ent
        self.text = ent.text
        self.start_char = ent.start_char
        self.end_char = ent.end_char
        self.vector = ent.vector
        self.vector_norm = ent.vector_norm
        
        # Linker is not carried over to keep the size down
        if linker is None:
            self.tui = None
            self.type = None
        else:
            tui = self.get_tui(rank=1, linker=linker)
            if len(tui) > 0:
                self.tui = tui[0]
            else:
                self.tui = ''
            self.type = self.get_type(rank=1, linker=linker)
        
    @property
    def cui(self):
        ''' The concept unique identifier (CUI) for the entity.
        '''
        return self.get_cui(rank=1, prob=False)
    
    @property
    def cuis(self):
        ''' All CUIs for the entity.
        '''
        return self.get_all_cuis(prob=False)
    
    def link(self, linker):
        ''' Link entity to a knowledge base.
        '''
        
        self.tui = self.get_tui(rank=1, linker=linker)[0]
        self.type = self.get_type(rank=1, linker=linker)
        
    def get_cui(self, rank=1, prob=False):
        ''' Retrieve n-th ranked CUI for the entity.
        '''
        
        cuis = self.entity._.kb_ents
        if len(cuis) > 0:
            cui = cuis[rank-1]
            if prob == False:
                return cui[0]
            else:
                return cui
        else:
            cui = []
            return cui
    
    def get_all_cuis(self, prob=False):
        ''' Retrieve all CUIs associated with the entity.
        '''
        
        cuis = dict(self.entity._.kb_ents)
        if len(cuis) > 0:
            if prob == False:
                return list(cuis.keys())
            else:
                return cuis
        else:
            return cuis
    
    def get_tui(self, linker, rank=1):
        ''' Retrieve the n-th ranked TUI for the entity.
        '''
        
        cui = self.get_cui(rank=rank, prob=False)
        if len(cui) > 0:
            tui = linker.kb.cui_to_entity[cui].types
        else:
            tui = []
        
        return tui
    
    def get_all_tuis(self, linker):
        ''' Retrieve all TUIs for associated with the entity.
        '''
        
        cuis = self.get_all_cuis(prob=False)
        if len(cuis) > 0:
            tuis = [linker.kb.cui_to_entity[cui].types[0] for cui in cuis]
        else:
            tuis = []
        
        return tuis
    
    def get_type(self, linker, rank=1):
        ''' Retrieve the semantic type of the n-th ranked TUI of the entity.
        '''
            
        tui = self.get_tui(rank=rank, linker=linker)
        if len(tui) > 0:
            tui = tui[0]
            ent_type = linker.kb.semantic_type_tree.get_canonical_name(tui)
        else:
            ent_type = ''
        
        return ent_type
        
    def get_all_types(self, linker):
        ''' Retrieve all semantic types associated with the entity.
        '''
        
        tuis = self.get_all_tuis(linker=linker)
        if len(tuis) > 0:
            ent_types = [linker.kb.semantic_type_tree.get_canonical_name(tui) for tui in tuis]
        else:
            ent_types = ['']
        
        return ent_types


class Annotation:
    ''' Paired text entries and their associated labels.
    The labels can be generic, specific, or empty string.
    If the associated entities follow the UMLS convention, the labels can be
    type or concept unique identifiers (TUIs or CUIs).
    '''
    
    def __init__(self, texts, labels, **kwargs):
        
        self.texts = texts
        self.labels = labels
        self.len = len(labels)
        self.boundaries = kwargs.pop("bchars", [None]*self.len)
        
        if len(self.texts) != self.len:
            raise ValueError("The number of text entries and labels should be the same!")
        
    def to_dict(self, keys="type"):
        
        if keys == "text":
            return dict(zip(self.texts, self.labels))
        elif keys == "type":
            return {"text":self.texts,
                    "labels":self.labels,
                    "boundaries":self.boundaries}
    
    def to_tuples(self, keys="type"):
        ''' Convert the entity annotation to tuples with the format
        (text, labels, boundaries).
        '''
        
        if keys == "text":
            return [(t, l, b) for (t, l, b) in zip(self.texts, self.labels, self.boundaries)]
        elif keys == "type":
            return [tuple(self.texts),
                    tuple(self.labels),
                    tuple(self.boundaries)]
        

class SpanText:
    ''' Text span class.
    '''
    
    def __init__(self, span):
        
        self.span = span
        self.tokens = []
        self.texts = []
        self.ent_types = []
        
        # Retrieve the text and entity types for all tokens
        for token in self.span:
            self.tokens.append(token)
            self.texts.append(token.text)
            self.ent_types.append(token.ent_type_)
            
        self.texts = np.array(self.texts)
        self.ent_types = np.array(self.ent_types)
        
    @property
    def ents(self):
        return self.span.ents
    
    def get_entities(self, form="entity", linker=None):
        ''' Retrieve all detected entities in the text span.
        '''
        
        if form == "text":
            return [ent.text for ent in self.ents]
        elif form == "entity": # Return in entity form
            return [Entity(ent, linker=linker) for ent in self.ents]
    
    def get_annotation(self):
        ''' Generate paired annotations for every token text.
        '''
        
        return Annotation(texts=self.texts, labels=self.ent_types)


class UMLSText(SpanText):
    ''' Text span class with UMLS annotation.
    '''
    
    def __init__(self, span, linker=None):
        
        super(UMLSText, self).__init__(span=span)
        self.get_types(linker=linker)
        
    def get_types(self, linker):
        
        entities = self.get_entities(form="entity", linker=linker)
        self.tuis = [ent.tui for ent in entities]
        self.cuis = [ent.cui for ent in entities]
        # Use text boundaries as key to avoid repeats in text
        self.ent_chars = {(ent.start_char, ent.end_char):ent.text for ent in entities}
        self.ent_texts = list(self.ent_chars.values())
    
    def get_umls_annotation(self):
        ''' Replace generic entity labels with TUIs.
        '''
        
        # Obtain entity boundaries in text
        boundaries = list(self.ent_chars.keys())
        return Annotation(texts=self.ent_texts, labels=self.tuis, bchars=boundaries)
    
    @classmethod
    def from_text(self, text, pipeline, linker=None):
        
        ner_text = pipeline(text)
        return self(ner_text, linker=linker)
    
    @property
    def annotation(self):
        
        anno = self.get_umls_annotation()
        return anno.to_dict()
    

# def umls_annotate(text, pipeline, linker=None):
#     ''' Generate UMLS entity annotations.
#     '''
    
#     ner_text = pipeline(text)
#     umls_text = UMLSText(ner_text, linker=linker)
#     umls_anno = umls_text.get_umls_annotation()
    
#     return umls_anno
    
class QAnnotator(q.Question):
    
    def __init__(self, **kwargs):
        super(QAnnotator, self).__init__(**kwargs)
        
    def annotate(self, pipeline, part='options', linker=None):
        ''' Annotate selected text with UMLS entity tags.
        '''
        
        # Annotate the options in the question text.
        if part == 'options':
            text_annos = copy(self.options)
            for opt_id, option in self.options.items():            
                try:
                    anno = UMLSText.from_text(option,
                                              pipeline=pipeline,
                                              linker=linker)
                    text_annos[opt_id] = anno.annotation
                except:
                    text_annos[opt_id] = None
                    
        # Annotate the question context.
        elif part == 'question':
            text_annos = {}
            try:
                anno = UMLSText.from_Text(self.question,
                                          pipeline=pipeline,
                                          linker=linker)
                text_annos['question'] = anno.annotation
            except:
                text_annos['question'] = None
        
        else:
            raise NotImplementedError
            
        self.annotation = text_annos
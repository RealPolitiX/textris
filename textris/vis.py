from . import utils
import numpy as np
import matplotlib.cm as mplcm
import matplotlib.colors as mplc
from IPython.core.display import display, HTML


def highlight_text(text, cv, cmap='bwr', cmin=-1, cmax=1, normalize='midpoint'):
    
    colormap = getattr(mplcm, cmap)
    
    if normalize == 'midpoint':
        # Normalize the colormap with respect to the midpoint
        norm = utils.MidpointNormalize(vmin=cmin, vmax=cmax, midpoint=0)
    else:
        norm = mplc.Normalize(vmin=cmin, vamx=cmax)
    mapper = mplcm.ScalarMappable(norm=norm, cmap=colormap)
    rgba = mapper.to_rgba(cv)
    color = mplc.rgb2hex(rgba, keep_alpha=True)
    
    # Construct the text with background color
    text_decor = '<span style="background-color:' +color+ '">' +text+ '</span>'
    
    return text_decor


class ValuedText:
    # Class for text with associated values
    
    def __init__(self, text_segs, values):
        
        if type(text_segs) != str:
            self.text_segs = text_segs
            self.text = ''.join(text_segs)
        else:
            self.text_segs = None
            self.text = text_segs
        self.values = values
        
    def highlight(self, disp=True, cmin=None, cmax=None, selector=None, **kwargs):
        # Highlight the text and display in html
        
        if cmin is None:
            cmin = self.values.min()
        if cmax is None:
            cmax = self.values.max()
        normscale = kwargs.pop('normalize', 'midpoint')
        cmap = kwargs.pop('cmap', 'bwr')
        
        hili_kwargs = {'cmin':cmin, 'cmax':cmax,
                       'normalize':normscale, 'cmap':cmap}
        
        modtext_segs = [highlight_text(word, c, **hili_kwargs) for word, c in zip(self.text_segs, self.values)]
        if selector is not None:
            a, b = selector
            modtext_segs = modtext_segs[a:b]
        text = ''.join(modtext_segs)
        text_hilite = text.replace('\n', '<br>')
        
        if disp:
            display(HTML(text_hilite))
        else:
            return text_hilite
raw = open('paper.tex', 'rb').read().decode('utf-8')
crlf = '\r\n' in raw
src = raw.replace('\r\n', '\n')

def replace_figure(label, fragfile):
    global src
    li = src.index('\\label{' + label + '}')
    b = src.rfind('\\begin{figure', 0, li)
    e = src.index('\\end{figure', li)
    e = src.index('}', e) + 1
    src = src[:b] + open(fragfile, encoding='utf-8').read() + src[e:]

for lab, f in [('fig:mainiid', 'fig_mainiid.tex'),
               ('fig:mainnoniid', 'fig_mainnoniid.tex'),
               ('fig:tierablation', 'fig_tierablation.tex')]:
    replace_figure(lab, f)

# fig:clientdist — replace if present, else insert right after fig:mainnoniid
if '\\label{fig:clientdist}' in src:
    replace_figure('fig:clientdist', 'fig_clientdist.tex')
else:
    li = src.index('\\label{fig:mainnoniid}')
    e = src.index('\\end{figure*}', li) + len('\\end{figure*}')
    src = (src[:e] + '\n\n' +
           open('fig_clientdist.tex', encoding='utf-8').read() + src[e:])

# fig:clientseeds — replace if present, else insert right after fig:clientdist
if '\\label{fig:clientseeds}' in src:
    replace_figure('fig:clientseeds', 'fig_clientseeds.tex')
else:
    li = src.index('\\label{fig:clientdist}')
    e = src.index('\\end{figure*}', li) + len('\\end{figure*}')
    src = (src[:e] + '\n\n' +
           open('fig_clientseeds.tex', encoding='utf-8').read() + src[e:])

# fig:partition — replace if present, else insert right after fig:clientseeds
if '\\label{fig:partition}' in src:
    replace_figure('fig:partition', 'fig_partition.tex')
else:
    li = src.index('\\label{fig:clientseeds}')
    e = src.index('\\end{figure*}', li) + len('\\end{figure*}')
    src = (src[:e] + '\n\n' +
           open('fig_partition.tex', encoding='utf-8').read() + src[e:])

src = src.replace('\\usepgfplotslibrary{groupplots}\n\\usepgfplotslibrary{fillbetween}',
                  '\\usepgfplotslibrary{groupplots}')
out = src.replace('\n', '\r\n') if crlf else src
open('paper.tex', 'wb').write(out.encode('utf-8'))
print('spliced')

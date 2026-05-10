interface SilentStructureViewerProps {
  pdbContent?: string
  secondPdbContent?: string
  title?: string
  heightClassName?: string
  emptyText?: string
}

const escapeTemplate = (value: string): string =>
  value.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$/g, '\\$')

export function SilentStructureViewer({
  pdbContent,
  secondPdbContent,
  title = 'Structure preview',
  heightClassName = 'h-64',
  emptyText = '暂无结构数据',
}: SilentStructureViewerProps) {
  if (!pdbContent) {
    return (
      <div className={`${heightClassName} flex items-center justify-center bg-gray-50 text-sm text-gray-400`}>
        {emptyText}
      </div>
    )
  }

  const first = escapeTemplate(pdbContent)
  const second = secondPdbContent ? escapeTemplate(secondPdbContent) : ''
  const srcDoc = `<!DOCTYPE html>
<html>
<head>
  <script>
    var silent=function(){};
    console.log=silent;console.warn=silent;console.info=silent;console.debug=silent;
  </script>
  <script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script>
  <style>
    body{margin:0;padding:0;overflow:hidden;background:#f8fafc}
    #app{width:100vw;height:100vh}
    #error{display:none;position:absolute;inset:0;align-items:center;justify-content:center;font:12px sans-serif;color:#64748b;background:#f8fafc}
  </style>
</head>
<body>
  <div id="app"></div>
  <div id="error">结构预览加载失败</div>
  <script>
    window.onerror=function(){document.getElementById('error').style.display='flex';return true;};
    molstar.Viewer.create('app',{layoutIsExpanded:false,layoutShowControls:false}).then(function(v){
      var pdb1=\`${first}\`;
      var pdb2=\`${second}\`;
      if (pdb2) {
        Promise.all([v.loadStructureFromData(pdb1,'pdb'),v.loadStructureFromData(pdb2,'pdb')]).then(function(loaded){
          try {
            v.visual.update({structure:loaded[0]},{type:'cartoon',color:{r:245,g:158,b:11},opacity:0.7});
            v.visual.update({structure:loaded[1]},{type:'cartoon',color:{r:59,g:130,b:246},opacity:0.7});
          } catch (e) {}
        });
      } else {
        v.loadStructureFromData(pdb1,'pdb');
      }
    }).catch(function(){document.getElementById('error').style.display='flex';});
  </script>
</body>
</html>`

  return (
    <iframe
      title={title}
      srcDoc={srcDoc}
      className={`w-full ${heightClassName} border-0 bg-gray-50`}
      sandbox="allow-scripts allow-same-origin"
    />
  )
}

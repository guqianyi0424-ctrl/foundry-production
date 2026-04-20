interface StepSection { title: string; icon: string; steps: string[]; items?: undefined; faqs?: undefined }
interface ItemSection { title: string; icon: string; items: Array<{ label: string; desc: string }>; steps?: undefined; faqs?: undefined }
interface FaqSection { title: string; icon: string; faqs: Array<{ q: string; a: string }>; steps?: undefined; items?: undefined }
type Section = StepSection | ItemSection | FaqSection

export function HelpPage() {
  const sections: Section[] = [
    {
      title: '快速开始',
      icon: '🚀',
      steps: [
        '上传目标蛋白结构文件（PDB/CIF格式）',
        '查看左侧序列视图和右侧3D结构，点击残基选择热点',
        '或点击"预测热点"按钮，使用DL模型自动选取Top-3热点',
        '调整Binder长度（默认80aa），点击"运行"开始设计',
        '在结果Tab页中查看RFD3/MPNN/RF3的详细结果',
        '筛选RMSD < 2.0Å 且 pLDDT > 80 的最佳设计',
      ],
    },
    {
      title: '参数说明',
      icon: '⚙️',
      items: [
        { label: 'Binder长度', desc: '设计的结合蛋白长度，推荐60-100aa' },
        { label: '热点残基', desc: '靶标表面关键残基，决定Binder的结合位置和方向' },
        { label: 'RMSD阈值', desc: 'RF3验证的通过标准，默认< 2.0Å' },
        { label: 'pLDDT', desc: '预测置信度分数，>85为高置信度' },
      ],
    },
    {
      title: '结果解读',
      icon: '📊',
      items: [
        { label: 'RFD3主链', desc: '生成的主链骨架结构，pLDDT越高越可靠' },
        { label: 'MPNN序列', desc: '设计的氨基酸序列，得分越高与主链兼容性越好' },
        { label: 'RF3验证', desc: '验证设计结构的稳定性，RMSD<2.0Å为通过' },
        { label: 'PAE热图', desc: '预测对齐误差，低值区域表示结构可靠' },
      ],
    },
    {
      title: '常见问题',
      icon: '❓',
      faqs: [
        { q: '上传PDB后无法解析？', a: '检查文件格式是否正确，确保包含ATOM记录，编码为UTF-8' },
        { q: '热点预测失败？', a: '确保Python环境已安装biotite和模型权重，检查GPU可用性' },
        { q: 'RFD3运行超时？', a: '正常需10-30分钟，检查foundry环境配置和GPU内存' },
        { q: '所有设计RMSD都很高？', a: '尝试更换热点残基、调整Binder长度、确认靶标结构质量' },
      ],
    },
  ]

  return (
    <div className="p-6 max-w-[900px] mx-auto">
      <h1 className="text-xl font-bold text-gray-900 mb-6">帮助</h1>

      <div className="space-y-5">
        {sections.map((sec) => (
          <section key={sec.title} className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-base font-bold text-gray-900 flex items-center gap-2 mb-4">
              <span>{sec.icon}</span>{sec.title}
            </h2>

            {sec.steps && (
              <ol className="space-y-2.5">
                {sec.steps.map((step, i) => (
                  <li key={i} className="flex gap-3 text-sm text-gray-700">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-primary-100 text-primary-700 text-xs font-bold flex items-center justify-center mt-0.5">{i + 1}</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            )}

            {sec.items && (
              <div className="grid grid-cols-2 gap-4">
                {sec.items.map((item) => (
                  <div key={item.label} className="bg-gray-50 rounded-lg p-4">
                    <div className="text-xs font-semibold text-primary-700 mb-1">{item.label}</div>
                    <div className="text-sm text-gray-600">{item.desc}</div>
                  </div>
                ))}
              </div>
            )}

            {sec.faqs && (
              <div className="space-y-4">
                {sec.faqs.map((faq, i) => (
                  <div key={i} className="border-l-2 border-primary-200 pl-4">
                    <div className="text-sm font-semibold text-gray-900 mb-1">Q: {faq.q}</div>
                    <div className="text-sm text-gray-600">A: {faq.a}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        ))}

        <section className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-xl border border-primary-100 p-6">
          <h2 className="text-base font-bold text-primary-800 mb-2">📖 详细文档</h2>
          <p className="text-sm text-primary-700 mb-3">完整的蛋白质Binder设计指南，包括靶标选择策略、参数优化建议和实验验证流程。</p>
          <a href="/docs/binder_design_guide.md" target="_blank" className="inline-flex items-center gap-1.5 text-sm font-medium text-primary-600 hover:text-primary-800 transition-colors">
            查看完整指南 →
          </a>
        </section>
      </div>
    </div>
  )
}

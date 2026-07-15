// Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
// 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
// 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

(function(){
  // 注入导航样式，保证所有页面一致（即使各页 style 未含 .nav 规则）
  if(!document.getElementById('nav-style')){
    var s = document.createElement('style');
    s.id = 'nav-style';
    s.textContent = '.nav{position:sticky;top:0;background:rgba(15,23,42,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:14px;flex-wrap:wrap;z-index:10}.nav a{font-size:13px;color:var(--sub)}.nav a:hover{color:var(--accent)}.nav .home{color:var(--accent);font-weight:700}';
    document.head.appendChild(s);
  }
  var main = [
    {t:'🏠 首页', h:'index.html', home:true},
    {t:'01 缘起', h:'01-origin.html'},
    {t:'02 价值', h:'02-value.html'},
    {t:'03 目的', h:'03-goal.html'},
    {t:'04 计划', h:'04-plan.html'},
    {t:'05 路径', h:'05-path.html'},
    {t:'06 开发', h:'06-dev.html'},
    {t:'07 测试', h:'07-test.html'},
    {t:'08 实施', h:'08-implement.html'},
    {t:'09 反馈', h:'09-feedback.html'},
    {t:'10 改进', h:'10-improve.html'}
  ];
  var supp = [
    {t:'术语表', h:'11-glossary.html'},
    {t:'风险登记', h:'12-risk.html'},
    {t:'WBS/RACI', h:'13-wbs.html'},
    {t:'数据字典', h:'14-datadict.html'},
    {t:'采集&算法清单', h:'data-algo-checklist.html'},
    {t:'试点/选型', h:'15-pilot.html'},
    {t:'培训/FAQ', h:'16-training.html'},
    {t:'验收/立项', h:'17-acceptance.html'}
  ];
  var h = '';
  main.forEach(function(l,i){
    if(i===1) h += '<span style="color:var(--line);margin:0 4px">|</span>';
    h += '<a'+(l.home?' class="home"':'')+' href="'+l.h+'">'+l.t+'</a>';
  });
  h += '<span style="color:var(--line);margin:0 4px">|</span><span style="color:var(--sub);font-size:12px;align-self:center">补充:</span>';
  supp.forEach(function(l){ h += '<a href="'+l.h+'">'+l.t+'</a>'; });
  var el = document.getElementById('nav');
  if(el) el.innerHTML = h;
})();

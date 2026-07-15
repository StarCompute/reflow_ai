<?php
/* Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
   许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

   demo.php —— 网页版「两张表拿推荐」，对应 docs/EASY.html 的操作。
   无需懂 AI：选好历史数据 + 新任务，点一下按钮，系统给出推荐参数。

   运行方式（在本目录起一个 PHP 服务即可）：
       php -S localhost:8000
   然后浏览器打开 http://localhost:8000/demo.php
*/
$root   = dirname(dirname(__DIR__));          // 项目根（含 proctune 包）
$demoDir = __DIR__;                            // 本文件所在目录
$resultFile = $demoDir . '/recommend_result.csv';

// 默认示例文件（与页面链接对应）
$defHistory = 'history_datas.csv';
$defInput   = 'input.csv';

$msg = '';
$json = null;
$rows = [];
$cols = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $knob    = trim($_POST['knob_cols'] ?? '');
    $context = trim($_POST['context_cols'] ?? '');
    $quality = trim($_POST['quality_col'] ?? '');
    $topk    = intval($_POST['top_k'] ?? 1);
    $ntrees  = intval($_POST['n_trees'] ?? 200);

    $args = [
        '--knob-cols' => $knob,
        '--context-cols' => $context,
        '--quality-col' => $quality,
        '--top-k' => $topk,
        '--n-trees' => $ntrees,
        '--output' => escapeshellarg($resultFile),
    ];

    // 历史样本来源
    $hsrc = $_POST['history_source'] ?? 'csv';
    if ($hsrc === 'mysql') {
        $args['--history-source'] = 'mysql';
        $args['--history-table'] = escapeshellarg($_POST['history_table'] ?? '');
        $args['--mysql-host'] = escapeshellarg($_POST['mysql_host'] ?? '127.0.0.1');
        $args['--mysql-user'] = escapeshellarg($_POST['mysql_user'] ?? 'root');
        $args['--mysql-pass'] = escapeshellarg($_POST['mysql_pass'] ?? '');
        $args['--mysql-db']   = escapeshellarg($_POST['mysql_db'] ?? '');
        $args['--mysql-port'] = intval($_POST['mysql_port'] ?? 3306);
    } else {
        $args['--history-source'] = 'csv';
        $args['--history-csv'] = escapeshellarg($demoDir . '/' . ($_POST['history_csv'] ?? $defHistory));
    }

    // 新任务来源
    $isrc = $_POST['input_source'] ?? 'csv';
    if ($isrc === 'mysql') {
        $args['--input-source'] = 'mysql';
        $args['--input-table'] = escapeshellarg($_POST['input_table'] ?? '');
        $args['--mysql-host'] = escapeshellarg($_POST['mysql_host'] ?? '127.0.0.1');
        $args['--mysql-user'] = escapeshellarg($_POST['mysql_user'] ?? 'root');
        $args['--mysql-pass'] = escapeshellarg($_POST['mysql_pass'] ?? '');
        $args['--mysql-db']   = escapeshellarg($_POST['mysql_db'] ?? '');
        $args['--mysql-port'] = intval($_POST['mysql_port'] ?? 3306);
    } else {
        $args['--input-source'] = 'csv';
        $args['--input-csv'] = escapeshellarg($demoDir . '/' . ($_POST['input_csv'] ?? $defInput));
    }

    $cmd = 'cd ' . escapeshellarg($root) . ' && python -m proctune.easy.web_bridge';
    foreach ($args as $k => $v) {
        $cmd .= ' ' . $k . ' ' . $v;
    }
    $cmd .= ' 2> ' . escapeshellarg($demoDir . '/_php_err.log');

    exec($cmd, $out, $rc);
    $last = trim(end($out));
    $json = json_decode($last, true);

    if ($rc !== 0 || !$json || !($json['ok'] ?? false)) {
        $err = $json['error'] ?? '';
        if (!$err && file_exists($demoDir . '/_php_err.log')) {
            $err = file_get_contents($demoDir . '/_php_err.log');
        }
        $msg = '执行失败（请确认 python 在 PATH 中、proctune 可导入、依赖已安装）：' . htmlspecialchars($err);
    } else {
        $msg = '推荐完成：' . htmlspecialchars($json['summary']);
        if (file_exists($resultFile)) {
            if (($fh = fopen($resultFile, 'r')) !== false) {
                while (($r = fgetcsv($fh)) !== false) {
                    if (empty($cols)) { $cols = $r; continue; }
                    $rows[] = $r;
                }
                fclose($fh);
            }
        }
    }
}
?>
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>工艺参数推荐 · 网页演示（proctune）</title>
<style>
  :root{--bg:#fff;--fg:#1f2328;--accent:#2563eb;--border:#d0d7de;--code:#f6f8fa;}
  *{box-sizing:border-box;}
  body{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;
       background:#f5f7fa;color:var(--fg);line-height:1.7;}
  .wrap{max-width:960px;margin:0 auto;padding:32px 20px 60px;}
  h1{font-size:24px;border-bottom:2px solid var(--border);padding-bottom:12px;}
  .card{background:#fff;border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin:18px 0;}
  label{display:block;font-weight:600;margin:12px 0 4px;}
  input[type=text],input[type=number]{width:100%;padding:8px 10px;border:1px solid var(--border);
       border-radius:6px;font-size:14px;}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
  .row{display:flex;gap:18px;align-items:center;margin:8px 0;}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:11px 22px;
         font-size:15px;font-weight:700;cursor:pointer;}
  button:hover{opacity:.92;}
  .msg{background:#eafaf0;border-left:4px solid #2da44e;padding:10px 14px;border-radius:0 6px 6px 0;margin:14px 0;}
  .err{background:#fff0f0;border-left:4px solid #cf222e;}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;}
  th,td{border:1px solid var(--border);padding:8px 10px;text-align:left;}
  th{background:var(--code);}
  .links a{color:var(--accent);margin-right:16px;}
  code{background:var(--code);padding:2px 6px;border-radius:5px;}
  .muted{color:#57606a;font-size:13px;}
  details.tech{background:#f0f6ff;border:1px solid #bcd4f6;border-radius:10px;padding:14px 18px;margin:18px 0;}
  details.tech summary{font-weight:700;cursor:pointer;color:var(--accent);font-size:15px;}
  details.tech table{margin:12px 0;}
  details.tech th{background:#e3edfb;white-space:nowrap;}
  details.tech .flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0;font-size:14px;}
  details.tech .flow b{background:#fff;border:1px solid #bcd4f6;border-radius:8px;padding:6px 12px;}
  details.tech .flow span{color:var(--accent);font-weight:700;}
</style>
</head>
<body>
<div class="wrap">
  <h1>工艺参数推荐 · 网页演示</h1>
  <p class="muted">版权归 <strong>蒲俊杰（Pu Junjie）</strong>。个人使用（含个人商业）免费；企业商业使用需付费授权，详见 LICENSE.md。</p>

  <div class="card">
    <p>本页对应 <code>docs/EASY.html</code> 的操作：给系统两张表（历史样本 + 新任务），它算出该用什么参数。
       默认已带示例文件，直接点「开始推荐」即可。</p>
    <p class="links">
      <a href="history_datas.csv" download>下载历史样本 history_datas.csv</a>
      <a href="input.csv" download>下载新任务 input.csv</a>
      <?php if (file_exists($resultFile)): ?><a href="recommend_result.csv" download>下载推荐结果 recommend_result.csv</a><?php endif; ?>
    </p>
  </div>

  <details class="tech">
    <summary>它背后是什么算法？（点击展开技术说明）</summary>
    <p>本系统在工业上称为 <strong>「受控工艺参数推荐」（Controlled Process Parameter Recommendation）</strong>，
       核心算法是 <strong>贝叶斯优化（Bayesian Optimization, BO）</strong>——自实现的 <strong>GP + UCB</strong>（高斯过程代理模型 + 上置信界采集函数）。</p>
    <div class="flow">
      <b>历史数据</b><span>→</span><b>代理模型（学参数↔质量规律）</b><span>→</span><b>贝叶斯优化（寻最优）</b><span>→</span><b>推荐参数</b>
    </div>
    <table>
      <thead><tr><th>维度</th><th>说明</th></tr></thead>
      <tbody>
        <tr><td>核心算法</td><td>贝叶斯优化 BO（GP + UCB）</td></tr>
        <tr><td>模型层</td><td>代理模型 + 质量模型 + 推荐引擎（BO + 离散枚举）</td></tr>
        <tr><td>问题本质</td><td>黑盒/灰盒工艺参数寻优（无解析式，从数据学规律反求最优旋钮）</td></tr>
        <tr><td>工业别名</td><td>工艺参数优化、参数寻优、实验设计 DOE、代理模型辅助优化</td></tr>
      </tbody>
    </table>
    <p class="muted">连续旋钮走 BO 序贯寻优；离散旋钮走枚举择优。已在回流焊、注塑、冶炼等真实产线验证。</p>
  </details>

  <?php if ($msg): ?>
    <div class="msg <?php echo (strpos($msg,'失败')===0)?'err':''; ?>"><?php echo $msg; ?></div>
  <?php endif; ?>

  <form method="post">
    <div class="card">
      <h3>① 历史样本（训练用）</h3>
      <div class="row">
        <label style="margin:0;"><input type="radio" name="history_source" value="csv" <?php echo ($_POST['history_source']??'csv')=='csv'?'checked':''; ?>> CSV 文件</label>
        <label style="margin:0;"><input type="radio" name="history_source" value="mysql" <?php echo ($_POST['history_source']??'')=='mysql'?'checked':''; ?>> MySQL 表</label>
      </div>
      <label>CSV 文件名（位于本目录）</label>
      <input type="text" name="history_csv" value="<?php echo htmlspecialchars($_POST['history_csv']??$defHistory); ?>">
      <div class="grid">
        <div>
          <label>MySQL 表名（历史）</label>
          <input type="text" name="history_table" value="<?php echo htmlspecialchars($_POST['history_table']??''); ?>">
        </div>
      </div>
    </div>

    <div class="card">
      <h3>② 新任务（待推荐）</h3>
      <div class="row">
        <label style="margin:0;"><input type="radio" name="input_source" value="csv" <?php echo ($_POST['input_source']??'csv')=='csv'?'checked':''; ?>> CSV 文件</label>
        <label style="margin:0;"><input type="radio" name="input_source" value="mysql" <?php echo ($_POST['input_source']??'')=='mysql'?'checked':''; ?>> MySQL 表</label>
      </div>
      <label>CSV 文件名（位于本目录）</label>
      <input type="text" name="input_csv" value="<?php echo htmlspecialchars($_POST['input_csv']??$defInput); ?>">
      <div class="grid">
        <div>
          <label>MySQL 表名（新任务）</label>
          <input type="text" name="input_table" value="<?php echo htmlspecialchars($_POST['input_table']??''); ?>">
        </div>
      </div>
    </div>

    <div class="card">
      <h3>③ 列配置（告诉系统哪几列是什么）</h3>
      <label>可调参数列（逗号分隔，系统要反推的）</label>
      <input type="text" name="knob_cols" value="<?php echo htmlspecialchars($_POST['knob_cols']??'料筒温度,注射压力,保压时间,模具温度'); ?>">
      <label>产品属性列（逗号分隔，已知不可调）</label>
      <input type="text" name="context_cols" value="<?php echo htmlspecialchars($_POST['context_cols']??'材料,壁厚'); ?>">
      <label>质量结果列</label>
      <input type="text" name="quality_col" value="<?php echo htmlspecialchars($_POST['quality_col']??'质量'); ?>">
      <div class="grid">
        <div><label>推荐条数 top_k</label><input type="number" name="top_k" value="<?php echo htmlspecialchars($_POST['top_k']??'1'); ?>"></div>
        <div><label>树数 n_trees（越大越准越慢）</label><input type="number" name="n_trees" value="<?php echo htmlspecialchars($_POST['n_trees']??'200'); ?>"></div>
      </div>
    </div>

    <div class="card">
      <h3>④ MySQL 连接（仅当上面选了 MySQL 时填写）</h3>
      <div class="grid">
        <div><label>主机</label><input type="text" name="mysql_host" value="<?php echo htmlspecialchars($_POST['mysql_host']??'127.0.0.1'); ?>"></div>
        <div><label>端口</label><input type="number" name="mysql_port" value="<?php echo htmlspecialchars($_POST['mysql_port']??'3306'); ?>"></div>
        <div><label>用户名</label><input type="text" name="mysql_user" value="<?php echo htmlspecialchars($_POST['mysql_user']??'root'); ?>"></div>
        <div><label>密码</label><input type="text" name="mysql_pass" value="<?php echo htmlspecialchars($_POST['mysql_pass']??''); ?>"></div>
        <div><label>数据库</label><input type="text" name="mysql_db" value="<?php echo htmlspecialchars($_POST['mysql_db']??''); ?>"></div>
      </div>
      <p class="muted">依赖：<code>pip install pymysql</code> 或 <code>pip install mysql-connector-python</code></p>
    </div>

    <button type="submit">开始推荐</button>
  </form>

  <?php if (!empty($rows)): ?>
  <div class="card">
    <h3>推荐结果（共 <?php echo count($rows); ?> 行）</h3>
    <p class="muted">「推荐_」开头的列就是系统给机台的建议参数；「预测良率」越大越好。</p>
    <table>
      <thead><tr><?php foreach ($cols as $c): ?><th><?php echo htmlspecialchars($c); ?></th><?php endforeach; ?></tr></thead>
      <tbody>
      <?php foreach ($rows as $r): ?>
        <tr><?php foreach ($r as $v): ?><td><?php echo htmlspecialchars($v); ?></td><?php endforeach; ?></tr>
      <?php endforeach; ?>
      </tbody>
    </table>
    <p><a href="recommend_result.csv" download>下载 recommend_result.csv</a></p>
  </div>
  <?php endif; ?>
</div>
</body>
</html>

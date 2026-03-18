---
layout: single
title: "搜索"
permalink: /search/
author_profile: false
---

<input id="site-search-input" type="search" placeholder="输入关键词（标题、标签、摘要）" style="width:100%;padding:0.75rem 0.9rem;font-size:1rem;border:1px solid #d1d5db;border-radius:8px;" />
<p id="site-search-count" style="margin-top:0.8rem;color:#6b7280;">输入关键词开始搜索。</p>
<ul id="site-search-results"></ul>

<script>
(function() {
  var docs = [
    {% for post in site.posts %}
    {
      title: {{ post.title | jsonify }},
      url: {{ post.url | relative_url | jsonify }},
      date: {{ post.date | date: "%Y-%m-%d" | jsonify }},
      tags: {{ post.tags | join: " " | jsonify }},
      content: {{ post.content | strip_html | strip_newlines | truncate: 220 | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ];

  var input = document.getElementById("site-search-input");
  var count = document.getElementById("site-search-count");
  var results = document.getElementById("site-search-results");

  function render(items) {
    results.innerHTML = "";
    if (!items.length) {
      count.textContent = "没有匹配结果。";
      return;
    }

    count.textContent = "找到 " + items.length + " 条结果。";
    items.forEach(function(item) {
      var li = document.createElement("li");
      li.style.marginBottom = "0.9rem";

      var a = document.createElement("a");
      a.href = item.url;
      a.textContent = item.title;
      a.style.fontWeight = "600";

      var meta = document.createElement("div");
      meta.textContent = item.date + (item.tags ? " | " + item.tags : "");
      meta.style.color = "#6b7280";
      meta.style.fontSize = "0.92rem";

      var excerpt = document.createElement("div");
      excerpt.textContent = item.content;
      excerpt.style.marginTop = "0.2rem";

      li.appendChild(a);
      li.appendChild(meta);
      li.appendChild(excerpt);
      results.appendChild(li);
    });
  }

  input.addEventListener("input", function(e) {
    var q = (e.target.value || "").trim().toLowerCase();
    if (!q) {
      results.innerHTML = "";
      count.textContent = "输入关键词开始搜索。";
      return;
    }

    var filtered = docs.filter(function(doc) {
      var haystack = (doc.title + " " + doc.tags + " " + doc.content).toLowerCase();
      return haystack.indexOf(q) !== -1;
    });

    render(filtered);
  });
})();
</script>

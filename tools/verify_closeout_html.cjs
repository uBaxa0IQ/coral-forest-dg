// Inspect only the generated offline document in a fresh headless browser.
const fs=require('fs'),path=require('path'),{pathToFileURL,fileURLToPath}=require('url');
const {chromium}=require('C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const root=path.resolve(__dirname,'..','reports','closeout'),review=path.join(root,'html_review');fs.mkdirSync(review,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.route(/^https?:/,r=>r.abort());await page.goto(pathToFileURL(path.join(root,'index.html')).href,{waitUntil:'load'});
 const images=await page.locator('figure img').evaluateAll(xs=>xs.map(x=>({src:x.getAttribute('src'),complete:x.complete,width:x.naturalWidth})));
 if(images.length!==4||images.some(x=>!x.complete||!x.width))throw Error('Missing plots');
 const links=await page.locator('a').evaluateAll(xs=>xs.map(x=>x.href));
 for(const href of links)if(href.startsWith('file:')&&!fs.existsSync(fileURLToPath(href)))throw Error('Broken file link '+href);
 const tables=await page.locator('table').count();if(tables!==4)throw Error('Missing tables');
 await page.screenshot({path:path.join(review,'overview.png')});
 await page.locator('table').first().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(review,'primary_table.png')});
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(review,'mobile.png')});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1))throw Error('Mobile overflow');
 if(errors.length)throw Error(errors.join('\n'));
 fs.writeFileSync(path.join(root,'html_validation.json'),JSON.stringify({offline:true,headless_isolated_browser:true,page_errors:errors,plots:images,tables,local_links_checked:links.length,mobile_no_overflow:true},null,2));
 await browser.close();console.log('Closeout HTML passed');
})().catch(e=>{console.error(e);process.exit(1)});

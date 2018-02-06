const exec = require('child_process').exec
function execp(cmd, opts = {}) {
    return new Promise((resolve, reject) => {
        const child = exec(cmd, opts, (err) => err ? reject(err) : resolve());
        child.stdout.pipe(process.stdout);
		child.stderr.pipe(process.stderr);
    });
}

const f = (async () => {
	try
	{ 
		if(process.platform === 'darwin')
		{
			try {
				await execp('brew install llvm')
			} catch(e) {
				await execp('rm -rf /usr/local/opt/llvm')
				await execp('brew install llvm')
			}
			await execp('env CXX="/usr/local/opt/llvm/bin/clang++ -L/usr/local/opt/llvm/lib" node-gyp rebuild')
		}
		else
		{
			await execp('node-gyp rebuild')
		}
	} catch(e)
	{
		console.error('failed to rebuild gyp', e)
		process.exit(1)
	}
})
f()

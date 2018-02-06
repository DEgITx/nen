const exec = require('child_process').exec
function execp(cmd, opts = {}) {
    return new Promise((resolve, reject) => {
        const child = exec(cmd, opts,(err) => err ? reject(err) : resolve());
        child.stdout.pipe(process.stdout);
		child.stderr.pipe(process.stderr);
    });
}

const f = (async () => {
	if(process.platform === 'darwin')
	{
		await execp('brew install llvm')
		await execp('env CXX="/usr/local/opt/llvm/bin/clang++ -L/usr/local/opt/llvm/lib" node-gyp rebuild')
	}
	else
	{
		await execp('node-gyp rebuild')
	}
})
f()
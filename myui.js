var http = require('http');
var url = require('url');
var fs = require('fs');

const hostname = '127.0.0.1';
const port = 8080;

const server = http.createServer(function(request, response) {
	var q = url.parse(request.url, true);
	var qdata = q.query;
	
	if(q.pathname == '/index' || q.pathname == '/index/'){
		if(!q.query.text1){
			fs.readFile('./index.html', function(err, data) {
				if (err) throw err;
				response.writeHead(200, {'Content-Type': 'text/html'});
				response.write(data);
				response.end();
			  });
		}
		else{
			const spawn = require("child_process").spawn;
			console.log( process.env.PATH );
			const pythonProcess = spawn('python',["./model.py",q.query.text1,q.query.text2,q.query.text3,q.query.text4,q.query.text5,q.query.text6,q.query.text7]).on('error', function( err ){ throw err });
			response.writeHeader(200);
			pythonProcess.stdout.on('data', (data) => {
				response.write(data);
				response.end();
			});
			
		}
	}
	else{
		fs.readFile('./index.html', function(err, data) {
			if (err) throw err;
			response.writeHead(200, {'Content-Type': 'text/html'});
			response.write(data);
			response.end();
		  });
	}
	
	
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
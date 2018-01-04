#include <node.h>
#include <node_object_wrap.h>
#include <v8.h>
#include "nen.hpp"
#include <uv.h>

using namespace v8;

class NeuralNetwork : public node::ObjectWrap
{
public:

	NeuralNetwork(unsigned inputs, unsigned outputs, unsigned layers, unsigned neurons)
	{
		network = new NEN::NeuronNetwork(inputs, outputs, layers, neurons, NEN::Adam);
	}

	~NeuralNetwork() {
		delete network;
	}

	static void Init(Local<Object> exports) {
	  Isolate* isolate = exports->GetIsolate();

	  // Prepare constructor template
	  Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
	  tpl->SetClassName(String::NewFromUtf8(isolate, "NeuralNetwork"));
	  tpl->InstanceTemplate()->SetInternalFieldCount(1);

	  Methods(tpl);

	  constructor.Reset(isolate, tpl->GetFunction());
	  exports->Set(String::NewFromUtf8(isolate, "NeuralNetwork"),
	               tpl->GetFunction());
	}

	static void Methods(Local<FunctionTemplate>& tpl)
	{
	  NODE_SET_PROTOTYPE_METHOD(tpl, "iterations", Iterations);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "forward", Forward);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "train", Train);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "backPropagate", BackProp);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "error", Error);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setAlgorithm", setAlgorithm);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setRate", setRate);
	}
private:
	static v8::Persistent<v8::Function> constructor;

	NEN::NeuronNetwork* network;

	static void New(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();

		if (args.IsConstructCall()) 
		{
			// Invoked as constructor: `new MyObject(...)`
			unsigned inputs = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
			unsigned outputs = args[1]->IsUndefined() ? 0 : args[1]->NumberValue();
			unsigned layers = args[2]->IsUndefined() ? 0 : args[2]->NumberValue();
			unsigned neurons = args[3]->IsUndefined() ? 0 : args[3]->NumberValue();

			NeuralNetwork* obj = new NeuralNetwork(inputs, outputs, layers, neurons);
			obj->Wrap(args.This());
			args.GetReturnValue().Set(args.This());
		} 
		else 
		{
			// Invoked as plain function `MyObject(...)`, turn into construct call.
			const int argc = 4;
			Local<Value> argv[argc] = { args[0], args[1], args[2], args[3] };
			Local<Context> context = isolate->GetCurrentContext();
			Local<Function> cons = Local<Function>::New(isolate, constructor);
			Local<Object> result = cons->NewInstance(context, argc, argv).ToLocalChecked();
			args.GetReturnValue().Set(result);
		}
	}

	static void Iterations(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* obj = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  args.GetReturnValue().Set(Number::New(isolate, obj->iterations));
	}

	static std::vector<double> toVector(Isolate* isolate, const Handle<Value>& value)
	{
	  Handle<Array> values = Handle<Array>::Cast(value);
	  if(!values->IsArray())
	  {
	  	isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments")));
	  }
	  std::vector<double> values_c;

	  int length = values->Length();
	  for(int i = 0; i < length; i++)
	  {
	  	values_c.push_back(values->Get(i)->NumberValue());
	  }

	  return values_c;
	}

	static std::vector<std::vector<double>> toVectorVector(Isolate* isolate, const Handle<Value>& value)
	{
	  if(!value->IsArray())
	  {
	  	isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments")));
	  }
	  Handle<Array> values = Handle<Array>::Cast(value);
	  std::vector<std::vector<double>> values_c;
	  int length = values->Length();
	  for(int i = 0; i < length; i++)
	  {
	  	values_c.push_back(toVector(isolate, values->Get(i)));
	  }
	  return values_c;
	}

	static Handle<Array> toArray(Isolate* isolate, const std::vector<double>& values)
	{
		Handle<Array> arr = Array::New(isolate);
	  	for(int i = 0; i < values.size(); i++)
	  		arr->Set(i, Number::New(isolate, values[i]));

	  	return arr;
	}

	static void Forward(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  std::vector<double> inputs_values = toVector(isolate, args[0]);
	  double* pointer = nullptr;

	  if(args[1]->IsNumber())
	  {
	  	pointer = (double*)((uintptr_t)args[1]->NumberValue());
	  }

	  if(pointer)
	  	network->forward(inputs_values, pointer);
	  else
	 	network->forward(inputs_values);
	  
	  auto outputs = network->output();
	  Handle<Array> out = toArray(isolate, outputs);

	  args.GetReturnValue().Set(out);
	}

	static void Error(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  std::vector<double> outputs_values = toVector(isolate, args[0]);

	  auto error = network->getError(outputs_values);

	  args.GetReturnValue().Set(Number::New(isolate, error));
	}

	static void Train(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  bool async = true;
	  double error_target = 0;

	  if(!args[2]->IsUndefined() && args[2]->IsObject())
	  {
	  	Handle<Object> options = Handle<Object>::Cast(args[2]);
	  	error_target = options->Get(String::NewFromUtf8(isolate, "error"))->NumberValue();
	  	async = !options->Get(String::NewFromUtf8(isolate, "sync"))->BooleanValue();
	  }

	  if(args[0]->IsArray())
	  {
	  	std::vector<double> errors;
	  	std::vector<std::vector<double>> inputs;
	  	std::vector<std::vector<double>> outputs;
	  	Local<Function> fitness;
	  	Local<Function> fitness_error;
	  	bool fitness_use = false;

	  	if(Handle<Array>::Cast(args[0])->Get(0)->IsArray())
	  	{
	  		inputs = toVectorVector(isolate, args[0]);
	  		if(args[1]->IsArray())
	  			outputs = toVectorVector(isolate, args[1]);
	  		else if(args[1]->IsObject())
	  		{
	  			Handle<Object> fitness_object = Handle<Object>::Cast(args[1]);
	  			fitness = Local<Function>::Cast(fitness_object->Get(String::NewFromUtf8(isolate, "fitness")));
	  			fitness_error = Local<Function>::Cast(fitness_object->Get(String::NewFromUtf8(isolate, "error")));
	  			fitness_use = true;
	  		}
	  	}
	  	else
	  	{
	  		inputs.push_back(toVector(isolate, args[0]));
	  		if(args[1]->IsArray())
	  			outputs.push_back(toVector(isolate, args[1]));
	  		else if(args[1]->IsObject())
	  		{
	  			Handle<Object> fitness_object = Handle<Object>::Cast(args[1]);
	  			fitness = Local<Function>::Cast(fitness_object->Get(String::NewFromUtf8(isolate, "fitness")));
	  			fitness_error = Local<Function>::Cast(fitness_object->Get(String::NewFromUtf8(isolate, "error")));
	  			fitness_use = true;
	  		}
	  	}

	  	if(async)
	  	{
	  		Persistent<Promise::Resolver, CopyablePersistentTraits<Promise::Resolver>> persistent;
	  		persistent.Reset(isolate, v8::Promise::Resolver::New(isolate));
	  		v8::Local<v8::Promise::Resolver> resolver = v8::Local<v8::Promise::Resolver>::New(isolate, persistent);
	  		args.GetReturnValue().Set(resolver->GetPromise());

  			uv_work_t * req = new uv_work_t;
  			struct ReqArgs
			{
				Persistent<Promise::Resolver, CopyablePersistentTraits<Promise::Resolver>> persistent;
				NEN::NeuronNetwork* network;
				std::vector<double> errors;
				std::vector<std::vector<double>> inputs;
				std::vector<std::vector<double>> outputs;
				double error_target;
				Persistent<Function, CopyablePersistentTraits<Function>> fitness;
				Persistent<Function, CopyablePersistentTraits<Function>> fitness_error;
				bool fitness_use = false;
			}* req_args = new ReqArgs;
			req->data = req_args;

  			req_args->persistent = persistent;
  			req_args->network = network;
  			req_args->inputs = toVectorVector(isolate, args[0]);
  			if(!fitness_use)
  				req_args->outputs = toVectorVector(isolate, args[1]);
  			req_args->error_target = error_target;

  			// fitness
  			if(fitness_use)
  			{
  				Persistent<Function, CopyablePersistentTraits<Function>> persistent_fitness;
  				persistent_fitness.Reset(isolate, fitness);
  				req_args->fitness = persistent_fitness;

  				Persistent<Function, CopyablePersistentTraits<Function>> persistent_fitness_error;
  				persistent_fitness_error.Reset(isolate, fitness_error);
  				req_args->fitness_error = persistent_fitness_error;

  				req_args->fitness_use = true;
  			}

  			uv_queue_work(uv_default_loop(), req, [](uv_work_t *req)
  			{
  				ReqArgs* data = (ReqArgs*)req->data;
  				if(!data->fitness_use)
  					data->errors = data->network->train(data->inputs, data->outputs, data->error_target);
  			}, [](uv_work_t *req, int status)
			{
				ReqArgs* data = (ReqArgs*)req->data;
				v8::Isolate* isolate = v8::Isolate::GetCurrent();
				v8::HandleScope scope(isolate);

				if(data->fitness_use)
				{
					Local<Function> fitness = Local<Function>::New(isolate, data->fitness);
  					Local<Function> fitness_error = Local<Function>::New(isolate, data->fitness_error);
  					auto network = data->network;
  					auto inputs = data->inputs;
  					auto context = isolate->GetCurrentContext()->Global();
			  		auto fitness_func = [&isolate, &context, &fitness, &fitness_error, &network, &inputs](unsigned long long iteration, unsigned i) -> std::pair<std::function<bool(double*, double*)>, std::function<double()>> {
						return std::pair<std::function<bool(double*, double*)>, std::function<double()>>(
						[&isolate, &context, &fitness, &network, i, iteration](double* c, double* d) -> bool {
							Handle<Value> argv[] = { 
								Number::New(isolate, (uintptr_t)c), 
								Number::New(isolate, (uintptr_t)d), 
								Number::New(isolate, i), 
								Number::New(isolate, iteration)
							};
							return fitness->Call(context, 4, argv)->BooleanValue();
						}, [&network, &context, &isolate, &fitness_error, i, iteration]() -> double {
							Handle<Value> argv[] = { 
								Number::New(isolate, i), 
								Number::New(isolate, iteration)
							};
							return fitness_error->Call(context, 2, argv)->NumberValue();
						});
					};
					data->errors = network->train(inputs, data->outputs, data->error_target, fitness_func);
				}

				v8::Local<v8::Promise::Resolver> local = v8::Local<v8::Promise::Resolver>::New(isolate, data->persistent);
				local->Resolve(toArray(isolate, data->errors));
				
				data->persistent.Reset();
				delete data;
			});
	  	}
	  	else
	  	{
			if(!fitness_use)
	  			errors = network->train(inputs, outputs, error_target);
	  		else
	  		{
	  			auto context = isolate->GetCurrentContext()->Global();
		  		auto fitness_func = [&isolate, &context, &fitness, &fitness_error, &network, &inputs](unsigned long long iteration, unsigned i) -> std::pair<std::function<bool(double*, double*)>, std::function<double()>> {
					return std::pair<std::function<bool(double*, double*)>, std::function<double()>>(
					[&isolate, &context, &fitness, &network, i, iteration](double* c, double* d) -> bool {
						Handle<Value> argv[] = { 
							Number::New(isolate, (uintptr_t)c), 
							Number::New(isolate, (uintptr_t)d), 
							Number::New(isolate, i), 
							Number::New(isolate, iteration)
						};
						return fitness->Call(context, 4, argv)->BooleanValue();
					}, [&network, &context, &isolate, &fitness_error, i, iteration]() -> double {
						Handle<Value> argv[] = { 
							Number::New(isolate, i), 
							Number::New(isolate, iteration)
						};
						return fitness_error->Call(context, 2, argv)->NumberValue();
					});
				};
	  			errors = network->train(inputs, outputs, error_target, fitness_func);
	  		}
	  		args.GetReturnValue().Set(toArray(isolate, errors));
	  	}
	  }
	}

	static void BackProp(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  std::vector<double> outputs = toVector(isolate, args[0]);
	  auto error = network->backPropagate(outputs);

	  args.GetReturnValue().Set(Number::New(isolate, error));
	}

	static void setAlgorithm(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->algorithm = (NEN::TrainingAlgorithm)((int)args[0]->NumberValue());
	}

	static void setRate(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->rate = args[0]->NumberValue();
	}
};

Persistent<Function> NeuralNetwork::constructor;

void init(Local<Object> exports) {
	NeuralNetwork::Init(exports);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)
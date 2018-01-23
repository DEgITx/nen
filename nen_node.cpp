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
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setActivation", setActivation);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setMoment", setMoment);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setDEpsilon", setDEpsilon);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setBeta1", setBeta1);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setBeta2", setBeta2);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setPopulation", setPopulation);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setElitePart", setElitePart);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setShuffle", setShuffle);
	  NODE_SET_PROTOTYPE_METHOD(tpl, "setMultiThreads", setMultiThreads);
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
	  if(args[1]->IsArray())
	  {
	  	std::vector<double> inputs_values = toVector(isolate, args[1]);
	  	double* pointer = nullptr;
	  	if(args[2]->IsNumber())
	    {
	  	   pointer = (double*)((uintptr_t)args[2]->NumberValue());
	    }
	    if(pointer)
	  	  network->forward(inputs_values, pointer);
	    else
	 	  network->forward(inputs_values);
	  }

	  auto error = network->getError(outputs_values);

	  args.GetReturnValue().Set(Number::New(isolate, error));
	}

	static void Train(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  bool async = true;
	  double error_target = 0;
	  Local<Function> iteration_callback;
	  bool iteration_callback_use = false;

	  if(!args[2]->IsUndefined() && args[2]->IsObject())
	  {
	  	Handle<Object> options = Handle<Object>::Cast(args[2]);
	  	error_target = options->Get(String::NewFromUtf8(isolate, "error"))->NumberValue();
	  	async = !options->Get(String::NewFromUtf8(isolate, "sync"))->BooleanValue();
	  	network->iterations_limit = (unsigned long long)options->Get(String::NewFromUtf8(isolate, "iterations"))->NumberValue();
	  	if(options->Get(String::NewFromUtf8(isolate, "iteration"))->IsFunction())
	  	{
	  		iteration_callback = Local<Function>::Cast(options->Get(String::NewFromUtf8(isolate, "iteration")));
	  		iteration_callback_use = true;
	  	}
	  	
	  }

	  if(args[0]->IsArray() || args[0]->IsFunction())
	  {
	  	std::vector<double> errors;
	  	std::vector<std::vector<double>> inputs;
	  	std::vector<std::vector<double>> outputs;
	  	Local<Function> fitness;
	  	Local<Function> fitness_error;
	  	bool fitness_use = false;

	  	if(args[0]->IsArray())
	  	{
	  		if(Handle<Array>::Cast(args[0])->Get(0)->IsArray())
		  	{
		  		inputs = toVectorVector(isolate, args[0]);
		  		outputs = toVectorVector(isolate, args[1]);
		  	}
		  	else
		  	{
		  		inputs.push_back(toVector(isolate, args[0]));
		  		outputs.push_back(toVector(isolate, args[1]));
		  	}
	  	}
	  	else
	  	{
  			fitness = Local<Function>::Cast(args[0]);
  			fitness_error = Local<Function>::Cast(args[1]);
  			fitness_use = true;
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
				Persistent<Function, CopyablePersistentTraits<Function>> iteration_callback;
				bool iteration_callback_use = false;
			}* req_args = new ReqArgs;
			req->data = req_args;

  			req_args->persistent = persistent;
  			req_args->network = network;
  			if(!fitness_use)
  			{
  				req_args->inputs = toVectorVector(isolate, args[0]);
  				req_args->outputs = toVectorVector(isolate, args[1]);
  			}
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

  			if(iteration_callback_use)
  			{
  				Persistent<Function, CopyablePersistentTraits<Function>> persistent_iteration_callback;
  				persistent_iteration_callback.Reset(isolate, iteration_callback);
  				req_args->iteration_callback = persistent_iteration_callback;

  				req_args->iteration_callback_use = true;
  			}

  			uv_queue_work(uv_default_loop(), req, [](uv_work_t *req)
  			{
  				ReqArgs* data = (ReqArgs*)req->data;
  				if(!data->fitness_use && !data->iteration_callback_use)
  					data->errors = data->network->train(data->inputs, data->outputs, data->error_target);
  			}, [](uv_work_t *req, int status)
			{
				ReqArgs* data = (ReqArgs*)req->data;
				v8::Isolate* isolate = v8::Isolate::GetCurrent();
				v8::HandleScope scope(isolate);

				Local<Function> iteration_callback;
				auto context = isolate->GetCurrentContext()->Global();

				if(data->iteration_callback_use)
				{
					iteration_callback = Local<Function>::New(isolate, data->iteration_callback);
					data->network->iteration_callback = [&isolate, &context, &iteration_callback](unsigned long long iteration, double error)
		  			{
		  				Handle<Value> argv[] = {
							Number::New(isolate, iteration),
							Number::New(isolate, error)
						};
						iteration_callback->Call(context, 2, argv);
		  			};
				}

				if(data->fitness_use)
				{
					Local<Function> fitness = Local<Function>::New(isolate, data->fitness);
  					Local<Function> fitness_error = Local<Function>::New(isolate, data->fitness_error);
  					auto network = data->network;
  					auto fitness_func = [&isolate, &context, &fitness, &fitness_error, &network](unsigned long long iteration) -> std::pair<std::function<bool(double*, double*)>, std::function<double()>> {
						return std::pair<std::function<bool(double*, double*)>, std::function<double()>>(
						[&isolate, &context, &fitness, &network, iteration](double* c, double* d) -> bool {
							Handle<Value> argv[] = { 
								Number::New(isolate, (uintptr_t)c), 
								Number::New(isolate, (uintptr_t)d), 
								Number::New(isolate, iteration)
							};
							return fitness->Call(context, 3, argv)->BooleanValue();
						}, [&network, &context, &isolate, &fitness_error, iteration]() -> double {
							Handle<Value> argv[] = { 
								Number::New(isolate, iteration)
							};
							return fitness_error->Call(context, 1, argv)->NumberValue();
						});
					};
					data->errors = network->train(data->inputs, data->outputs, data->error_target, fitness_func);
				}
				else if(data->iteration_callback_use)
				{
					data->errors = data->network->train(data->inputs, data->outputs, data->error_target);
				}

				v8::Local<v8::Promise::Resolver> local = v8::Local<v8::Promise::Resolver>::New(isolate, data->persistent);
				local->Resolve(toArray(isolate, data->errors));
				
				data->persistent.Reset();
				delete data;
			});
	  	}
	  	else
	  	{
	  		if(iteration_callback_use)
	  		{
	  			auto context = isolate->GetCurrentContext()->Global();
	  			network->iteration_callback = [&isolate, &context, &iteration_callback](unsigned long long iteration, double error)
	  			{
	  				Handle<Value> argv[] = {
						Number::New(isolate, iteration),
						Number::New(isolate, error)
					};
					iteration_callback->Call(context, 2, argv);
	  			};
	  		}

			if(!fitness_use)
	  			errors = network->train(inputs, outputs, error_target);
	  		else
	  		{
	  			auto context = isolate->GetCurrentContext()->Global();
		  		auto fitness_func = [&isolate, &context, &fitness, &fitness_error, &network](unsigned long long iteration) -> std::pair<std::function<bool(double*, double*)>, std::function<double()>> {
					return std::pair<std::function<bool(double*, double*)>, std::function<double()>>(
					[&isolate, &context, &fitness, &network, iteration](double* c, double* d) -> bool {
						Handle<Value> argv[] = { 
							Number::New(isolate, (uintptr_t)c), 
							Number::New(isolate, (uintptr_t)d),
							Number::New(isolate, iteration)
						};
						return fitness->Call(context, 3, argv)->BooleanValue();
					}, [&network, &context, &isolate, &fitness_error, iteration]() -> double {
						Handle<Value> argv[] = {
							Number::New(isolate, iteration)
						};
						return fitness_error->Call(context, 1, argv)->NumberValue();
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

	static void setActivation(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->activation = (NEN::ActivationFunction)((int)args[0]->NumberValue());
	}

	static void setMoment(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->momentum = args[0]->NumberValue();
	}

	static void setDEpsilon(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->d_epsilon = args[0]->NumberValue();
	}

	static void setBeta1(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->beta1 = args[0]->NumberValue();
	}

	static void setBeta2(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->beta2 = args[0]->NumberValue();
	}

	static void setPopulation(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->genetic_population_size = args[0]->NumberValue();
	}

	static void setElitePart(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->genetic_elite_part = args[0]->NumberValue();
	}

	static void setShuffle(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->enable_shuffle = args[0]->BooleanValue();
	}

	static void setMultiThreads(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  network->setMultiThreads(args[0]->BooleanValue());
	}
};

Persistent<Function> NeuralNetwork::constructor;

void init(Local<Object> exports) {
	NeuralNetwork::Init(exports);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)
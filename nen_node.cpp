#include <node.h>
#include <node_object_wrap.h>
#include <v8.h>
#include "nen.hpp"

using namespace v8;

class NeuralNetwork : public node::ObjectWrap
{
public:

	NeuralNetwork(unsigned inputs, unsigned outputs, unsigned layers, unsigned neurons)
	{
		network = new NEN::NeuronNetwork(inputs, outputs, layers, neurons, NEN::StochasticGradient);
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

	  network->forward(inputs_values);
	  auto outputs = network->output();
	  Handle<Array> out = toArray(isolate, outputs);

	  args.GetReturnValue().Set(out);
	}

	static void Train(const FunctionCallbackInfo<Value>& args) {
	  Isolate* isolate = args.GetIsolate();
	  NEN::NeuronNetwork* network = ObjectWrap::Unwrap<NeuralNetwork>(args.Holder())->network;

	  double error_target = 0;
	  if(!args[2]->IsUndefined() && args[2]->IsObject())
	  {
	  	Handle<Object> options = Handle<Object>::Cast(args[2]);
	  	error_target = options->Get(String::NewFromUtf8(isolate, "error"))->NumberValue();
	  }

	  if(args[0]->IsArray() && Handle<Array>::Cast(args[0])->Get(0)->IsArray())
	  {
	  	std::vector<double> errors;
	  	std::vector<std::vector<double>> inputs = toVectorVector(isolate, args[0]);
	  	std::vector<std::vector<double>> outputs = toVectorVector(isolate, args[1]);
	  	if(error_target > 0)
	  		errors = network->trainWhileError(inputs, outputs, 0, error_target);
	  	else
	  		errors = network->train(inputs, outputs);

	  	args.GetReturnValue().Set(toArray(isolate, errors));
	  }
	  else
	  {
	  	std::vector<double> inputs = toVector(isolate, args[0]);
	  	std::vector<double> outputs = toVector(isolate, args[1]);
	  	//if(error_target > 0)
	  	//	error = network->trainWhileError(inputs, outputs, 0, error_target);
	  	//else
	  	double error = network->train(inputs, outputs);

	  	args.GetReturnValue().Set(Number::New(isolate, error));
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
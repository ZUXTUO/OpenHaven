#include <arpa/inet.h>
#include <cmath>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <netinet/in.h>
#include <pthread.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>
#include <algorithm>

#include "rkllm.h"

LLMHandle g_chat_handle  = nullptr;
LLMHandle g_embed_handle = nullptr;

std::string g_chat_model_name  = "qwen-chat";
std::string g_embed_model_name = "text-embedding";

std::mutex g_chat_mutex;
std::mutex g_embed_mutex;

bool g_mock_embed = false;

enum class CtxType { CHAT, EMBED };

struct BaseContext {
    CtxType type;
    virtual ~BaseContext() = default;
};

struct InferContext : public BaseContext {
    InferContext() { type = CtxType::CHAT; }
    
    std::string accumulated_text;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
    bool error    = false;

    std::queue<std::string> token_queue;
    bool stream_mode = false;
    int client_fd    = -1;

    std::string utf8_pending;
};

struct EmbeddingContext : public BaseContext {
    EmbeddingContext() { type = CtxType::EMBED; }
    
    std::vector<float> embedding;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
    bool error    = false;
};

static size_t utf8_complete_len(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        size_t char_len;
        if      (c < 0x80) char_len = 1;
        else if (c < 0xC0) { ++i; continue; }
        else if (c < 0xE0) char_len = 2;
        else if (c < 0xF0) char_len = 3;
        else               char_len = 4;

        if (i + char_len > s.size()) break;
        i += char_len;
    }
    return i;
}

int rkllm_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    BaseContext* base_ctx = static_cast<BaseContext*>(userdata);
    if (!base_ctx) return 0;

    if (base_ctx->type == CtxType::EMBED) {
        EmbeddingContext* ectx = static_cast<EmbeddingContext*>(userdata);
        if (state == RKLLM_RUN_FINISH) {
            std::unique_lock<std::mutex> lock(ectx->mtx);
            ectx->finished = true;
            ectx->cv.notify_all();
        } else if (state == RKLLM_RUN_ERROR) {
            std::unique_lock<std::mutex> lock(ectx->mtx);
            ectx->error    = true;
            ectx->finished = true;
            ectx->cv.notify_all();
        } else if (state == RKLLM_RUN_NORMAL) {
            if (result && result->last_hidden_layer.hidden_states &&
                result->last_hidden_layer.embd_size > 0) {
                std::unique_lock<std::mutex> lock(ectx->mtx);
                const float* data = result->last_hidden_layer.hidden_states;
                int dim    = result->last_hidden_layer.embd_size;
                int ntok   = result->last_hidden_layer.num_tokens;
                int offset = (ntok > 1) ? (ntok - 1) * dim : 0;
                ectx->embedding.assign(data + offset, data + offset + dim);
            }
        }
        return 0;
    }

    if (base_ctx->type == CtxType::CHAT) {
        InferContext* ctx = static_cast<InferContext*>(userdata);
        if (state == RKLLM_RUN_FINISH) {
            std::unique_lock<std::mutex> lock(ctx->mtx);
            if (!ctx->utf8_pending.empty()) {
                if (ctx->stream_mode) {
                    ctx->token_queue.push(ctx->utf8_pending);
                    ctx->cv.notify_all();
                } else {
                    ctx->accumulated_text += ctx->utf8_pending;
                }
                ctx->utf8_pending.clear();
            }
            ctx->finished = true;
            ctx->cv.notify_all();
        } else if (state == RKLLM_RUN_ERROR) {
            std::unique_lock<std::mutex> lock(ctx->mtx);
            ctx->error = true;
            ctx->finished = true;
            ctx->cv.notify_all();
        } else if (state == RKLLM_RUN_NORMAL) {
            if (result && result->text) {
                std::unique_lock<std::mutex> lock(ctx->mtx);
                std::string chunk = ctx->utf8_pending + result->text;
                ctx->utf8_pending.clear();

                size_t complete = utf8_complete_len(chunk);
                ctx->utf8_pending = chunk.substr(complete);
                chunk = chunk.substr(0, complete);

                if (!chunk.empty()) {
                    if (ctx->stream_mode) {
                        ctx->token_queue.push(chunk);
                        ctx->cv.notify_all();
                    } else {
                        ctx->accumulated_text += chunk;
                    }
                }
            }
        }
        return 0;
    }

    return 0;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    while (++pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'));
    if (pos >= json.size()) return "";
    
    if (json[pos] == '"') {
        std::string val;
        ++pos;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                switch (json[pos]) {
                    case 'n': val += '\n'; break;
                    case 't': val += '\t'; break;
                    case 'r': val += '\r'; break;
                    case 'u': {
                        if (pos + 4 < json.size()) {
                            try {
                                std::string hex = json.substr(pos + 1, 4);
                                int cp = std::stoi(hex, nullptr, 16);
                                pos += 4;
                                if (cp < 0x80) val += (char)cp;
                                else if (cp < 0x800) { val += (char)(0xc0 | (cp >> 6)); val += (char)(0x80 | (cp & 0x3f)); }
                                else if (cp < 0x10000) { val += (char)(0xe0 | (cp >> 12)); val += (char)(0x80 | ((cp >> 6) & 0x3f)); val += (char)(0x80 | (cp & 0x3f)); }
                            } catch (...) {}
                        }
                        break;
                    }
                    default: val += json[pos];
                }
            } else {
                val += json[pos];
            }
            ++pos;
        }
        return val;
    }
    size_t end = json.find_first_of(",}\n\r ", pos);
    return json.substr(pos, end - pos);
}

static bool json_get_bool(const std::string& json, const std::string& key, bool def = false) {
    std::string v = json_get_string(json, key);
    if (v == "true") return true;
    if (v == "false") return false;
    return def;
}

struct ChatMessage { std::string role; std::string content; };

static std::vector<ChatMessage> json_get_messages(const std::string& json) {
    std::vector<ChatMessage> msgs;
    size_t arr_start = json.find("\"messages\"");
    if (arr_start == std::string::npos) return msgs;
    arr_start = json.find('[', arr_start);
    if (arr_start == std::string::npos) return msgs;

    size_t pos = arr_start + 1;
    int depth = 1;
    while (pos < json.size() && depth > 0) {
        size_t obj_start = json.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = obj_start + 1;
        int odepth = 1;
        while (obj_end < json.size() && odepth > 0) {
            if (json[obj_end] == '{') odepth++;
            else if (json[obj_end] == '}') odepth--;
            obj_end++;
        }
        std::string obj = json.substr(obj_start, obj_end - obj_start);
        ChatMessage msg;
        msg.role = json_get_string(obj, "role");
        msg.content = json_get_string(obj, "content");
        if (!msg.role.empty()) msgs.push_back(msg);
        pos = obj_end;
        if (json[pos] == ']') break;
    }
    return msgs;
}

static std::vector<std::string> json_get_input_strings(const std::string& json) {
    std::vector<std::string> result;
    std::string search = "\"input\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return result;
    while (++pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'));
    if (pos >= json.size()) return result;

    if (json[pos] == '"') {
        result.push_back(json_get_string(json, "input"));
    } else if (json[pos] == '[') {
        size_t i = pos + 1;
        while (i < json.size()) {
            while (i < json.size() && (json[i] == ' ' || json[i] == '\t' || json[i] == '\n' || json[i] == '\r')) ++i;
            if (i >= json.size() || json[i] == ']') break;
            if (json[i] == '"') {
                std::string val;
                ++i;
                while (i < json.size() && json[i] != '"') {
                    if (json[i] == '\\' && i + 1 < json.size()) {
                        ++i;
                        switch (json[i]) {
                            case 'n': val += '\n'; break;
                            case 't': val += '\t'; break;
                            case 'r': val += '\r'; break;
                            default:  val += json[i];
                        }
                    } else {
                        val += json[i];
                    }
                    ++i;
                }
                ++i;
                result.push_back(val);
            } else {
                while (i < json.size() && json[i] != ',' && json[i] != ']') ++i;
            }
            while (i < json.size() && (json[i] == ' ' || json[i] == ',')) ++i;
        }
    }
    return result;
}

static long long unix_timestamp() { return (long long)time(nullptr); }
static std::string gen_id(const std::string& prefix = "chatcmpl-") {
    static std::atomic<int> counter{0};
    return prefix + std::to_string(unix_timestamp()) + "-" + std::to_string(counter++);
}

struct HttpRequest {
    std::string method, path, version, body;
    std::map<std::string, std::string> headers;
    bool valid = false;
};

static HttpRequest parse_http_request(int fd) {
    HttpRequest req;
    std::string raw;
    char buf[4096];

    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) return req;
        buf[n] = '\0';
        raw += buf;
        if (raw.find("\r\n\r\n") != std::string::npos) break;
        if (raw.size() > 65536) return req;
    }

    size_t line_end = raw.find("\r\n");
    if (line_end == std::string::npos) return req;
    std::istringstream req_line(raw.substr(0, line_end));
    req_line >> req.method >> req.path >> req.version;

    size_t header_start = line_end + 2;
    size_t header_end = raw.find("\r\n\r\n");
    std::string headers_str = raw.substr(header_start, header_end - header_start);
    std::istringstream hss(headers_str);
    std::string hline;
    while (std::getline(hss, hline)) {
        if (!hline.empty() && hline.back() == '\r') hline.pop_back();
        size_t colon = hline.find(':');
        if (colon != std::string::npos) {
            std::string key = hline.substr(0, colon);
            std::string val = hline.substr(colon + 1);
            while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            req.headers[key] = val;
        }
    }

    size_t body_offset = header_end + 4;
    req.body = raw.substr(body_offset);
    int content_length = 0;
    if (req.headers.count("content-length")) {
        try { content_length = std::stoi(req.headers["content-length"]); } catch (...) {}
    }

    while ((int)req.body.size() < content_length) {
        ssize_t n = recv(fd, buf, std::min((int)sizeof(buf) - 1, content_length - (int)req.body.size()), 0);
        if (n <= 0) break;
        buf[n] = '\0';
        req.body += buf;
    }
    req.valid = true;
    return req;
}

static void send_response(int fd, int status, const std::string& content_type, const std::string& body, bool keep_alive = false) {
    std::string status_text = "OK";
    if (status == 400) status_text = "Bad Request";
    else if (status == 404) status_text = "Not Found";
    else if (status == 405) status_text = "Method Not Allowed";
    else if (status == 500) status_text = "Internal Server Error";
    else if (status == 503) status_text = "Service Unavailable";

    std::ostringstream resp;
    resp << "HTTP/1.1 " << status << " " << status_text << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
         << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
         << "Connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n\r\n"
         << body;
    std::string r = resp.str();
    send(fd, r.c_str(), r.size(), MSG_NOSIGNAL);
}

static void send_sse_headers(int fd) {
    std::string headers =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n";
    send(fd, headers.c_str(), headers.size(), MSG_NOSIGNAL);
}

static bool send_sse_event(int fd, const std::string& data) {
    std::string line = "data: " + data + "\n\n";
    return send(fd, line.c_str(), line.size(), MSG_NOSIGNAL) > 0;
}

static std::string build_prompt_from_messages(const std::vector<ChatMessage>& msgs) {
    std::string prompt;
    for (const auto& m : msgs) {
        if (m.role == "system") prompt += "[System]: " + m.content + "\n";
        else if (m.role == "user") prompt += m.content;
        else if (m.role == "assistant") prompt += "[Assistant]: " + m.content + "\n";
    }
    return prompt;
}

static std::string make_error_json(const std::string& type, const std::string& msg, const std::string& code = "") {
    std::string c = code.empty() ? type : code;
    return "{\"error\":{\"message\":\"" + json_escape(msg) + "\",\"type\":\"" + type + "\",\"code\":\"" + c + "\"}}";
}

static std::string handle_models() {
    long long ts = unix_timestamp();
    std::ostringstream j;
    j << "{\"object\":\"list\",\"data\":[";
    
    bool has_prev = false;
    if (g_chat_handle) {
        j << "{\"id\":\"" << json_escape(g_chat_model_name) << "\",\"object\":\"model\",\"created\":" << ts << ",\"owned_by\":\"rkllm\"}";
        has_prev = true;
    }
    if (g_embed_handle) {
        if (has_prev) j << ",";
        j << "{\"id\":\"" << json_escape(g_embed_model_name) << "\",\"object\":\"model\",\"created\":" << ts << ",\"owned_by\":\"rkllm\"}";
    }
    j << "]}";
    return j.str();
}

static bool do_chat_infer(const std::string& prompt, InferContext& ctx) {
    std::lock_guard<std::mutex> lock(g_chat_mutex);
    if (!g_chat_handle) return false;

    RKLLMInput input;
    memset(&input, 0, sizeof(RKLLMInput));
    input.input_type = RKLLM_INPUT_PROMPT;
    input.role = "user";
    input.prompt_input = (char*)prompt.c_str();

    RKLLMInferParam infer_params;
    memset(&infer_params, 0, sizeof(RKLLMInferParam));
    infer_params.mode = RKLLM_INFER_GENERATE;
    infer_params.keep_history = 0;

    int ret = rkllm_run(g_chat_handle, &input, &infer_params, &ctx);
    return (ret == 0 && !ctx.error);
}

static bool do_embed_infer(const std::string& text, std::vector<float>& out_vec) {
    std::lock_guard<std::mutex> lock(g_embed_mutex);
    if (!g_embed_handle) return false;

    if (g_mock_embed) {
        const int EMBED_DIM = 384;
        std::vector<float> embedding(EMBED_DIM);
        unsigned long hash = 5381;
        for (char c : text) hash = ((hash << 5) + hash) + (unsigned char)c;
        srand(hash);
        float norm = 0.0f;
        for (int i = 0; i < EMBED_DIM; ++i) {
            embedding[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            norm += embedding[i] * embedding[i];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-9f) { for (float& v : embedding) v /= norm; }
        out_vec = std::move(embedding);
        return true;
    } else {
        EmbeddingContext ectx;
        RKLLMInput input;
        memset(&input, 0, sizeof(RKLLMInput));
        input.input_type   = RKLLM_INPUT_PROMPT;
        input.role         = "user";
        input.prompt_input = (char*)text.c_str();

        RKLLMInferParam infer_params;
        memset(&infer_params, 0, sizeof(RKLLMInferParam));
        infer_params.mode         = RKLLM_INFER_GET_LAST_HIDDEN_LAYER;
        infer_params.keep_history = 0;

        int ret = rkllm_run(g_embed_handle, &input, &infer_params, &ectx);
        if (ret == 0) {
            std::unique_lock<std::mutex> ul(ectx.mtx);
            ectx.cv.wait_for(ul, std::chrono::seconds(60), [&]{ return ectx.finished; });
        }

        if (ret != 0 || ectx.error || ectx.embedding.empty()) return false;

        float norm = 0.0f;
        for (float v : ectx.embedding) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-9f) { for (float& v : ectx.embedding) v /= norm; }

        out_vec = std::move(ectx.embedding);
        return true;
    }
}

static std::string handle_chat_completions_sync(const std::string& body) {
    auto msgs = json_get_messages(body);
    if (msgs.empty()) return make_error_json("invalid_request_error", "No messages provided");

    std::string prompt = build_prompt_from_messages(msgs);
    std::string request_model = json_get_string(body, "model");
    if (request_model.empty()) request_model = g_chat_model_name;

    InferContext ctx;
    ctx.stream_mode = false;
    bool ok = do_chat_infer(prompt, ctx);

    if (!ok) return make_error_json("server_error", "Chat Inference failed");

    std::ostringstream j;
    j << "{\"id\":\"" << json_escape(gen_id("chatcmpl-")) << "\",\"object\":\"chat.completion\",\"created\":" << unix_timestamp()
      << ",\"model\":\"" << json_escape(request_model) << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" 
      << json_escape(ctx.accumulated_text) << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":-1,\"completion_tokens\":-1,\"total_tokens\":-1}}";
    return j.str();
}

static void handle_chat_completions_stream(int fd, const std::string& body) {
    auto msgs = json_get_messages(body);
    if (msgs.empty()) {
        send_response(fd, 400, "application/json", make_error_json("invalid_request_error", "No messages"));
        return;
    }

    std::string prompt = build_prompt_from_messages(msgs);
    std::string request_model = json_get_string(body, "model");
    if (request_model.empty()) request_model = g_chat_model_name;

    send_sse_headers(fd);

    InferContext ctx;
    ctx.stream_mode = true;
    ctx.client_fd = fd;
    std::string cid = gen_id("chatcmpl-");
    long long ts = unix_timestamp();

    std::atomic<bool> infer_started{false};
    std::thread infer_thread([&]() {
        infer_started = true;
        do_chat_infer(prompt, ctx);
    });

    while (!infer_started) std::this_thread::yield();

    bool client_ok = true;
    auto make_chunk = [&](const std::string& delta, bool is_last) -> std::string {
        std::ostringstream j;
        j << "{\"id\":\"" << json_escape(cid) << "\",\"object\":\"chat.completion.chunk\",\"created\":" << ts 
          << ",\"model\":\"" << json_escape(request_model) << "\",\"choices\":[{\"index\":0,\"delta\":{";
        if (is_last) j << "},\"finish_reason\":\"stop\"}]}";
        else j << "\"role\":\"assistant\",\"content\":\"" << json_escape(delta) << "\"},\"finish_reason\":null}]}";
        return j.str();
    };

    if (client_ok) client_ok = send_sse_event(fd, make_chunk("", false));

    while (client_ok) {
        std::unique_lock<std::mutex> lock(ctx.mtx);
        ctx.cv.wait_for(lock, std::chrono::milliseconds(100), [&]{ return !ctx.token_queue.empty() || ctx.finished; });
        while (!ctx.token_queue.empty()) {
            std::string token = ctx.token_queue.front();
            ctx.token_queue.pop();
            lock.unlock();
            client_ok = send_sse_event(fd, make_chunk(token, false));
            lock.lock();
        }
        if (ctx.finished) break;
    }

    if (client_ok) { send_sse_event(fd, make_chunk("", true)); send_sse_event(fd, "[DONE]"); }
    if (infer_thread.joinable()) infer_thread.join();
}

static std::string handle_embeddings(const std::string& body) {
    std::vector<std::string> inputs = json_get_input_strings(body);
    if (inputs.empty()) return make_error_json("invalid_request_error", "Field 'input' missing or empty");

    std::string request_model = json_get_string(body, "model");
    if (request_model.empty()) request_model = g_embed_model_name;

    std::ostringstream j;
    j << "{\"object\":\"list\",\"model\":\"" << json_escape(request_model) << "\",\"data\":[";

    int total_tokens = 0;
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        std::vector<float> vec;
        if (!do_embed_infer(inputs[idx], vec)) {
            return make_error_json("server_error", "Embedding inference failed for input " + std::to_string(idx));
        }
        
        int tok = 1; for (char c : inputs[idx]) if (c == ' ') ++tok; total_tokens += tok;

        if (idx) j << ",";
        j << "{\"object\":\"embedding\",\"index\":" << idx << ",\"embedding\":[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i) j << ",";
            char buf[32]; snprintf(buf, sizeof(buf), "%.8g", vec[i]);
            j << buf;
        }
        j << "]}";
    }

    j << "],\"usage\":{\"prompt_tokens\":" << total_tokens << ",\"total_tokens\":" << total_tokens << "}}";
    return j.str();
}

static void handle_request(int fd) {
    HttpRequest req = parse_http_request(fd);
    if (!req.valid) { close(fd); return; }

    if (req.method == "OPTIONS") { send_response(fd, 200, "text/plain", "", false); close(fd); return; }

    std::string path = req.path;
    size_t q = path.find('?'); if (q != std::string::npos) path = path.substr(0, q);

    if (req.method == "GET" && (path == "/v1/models" || path == "/v1/models/")) {
        send_response(fd, 200, "application/json", handle_models());
    } else if (req.method == "GET" && (path == "/v1/health" || path == "/health")) {
        send_response(fd, 200, "application/json", "{\"status\":\"ok\"}");
    } else if (req.method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions")) {
        if (!g_chat_handle) {
            send_response(fd, 503, "application/json", make_error_json("server_error", "Chat model not loaded"));
        } else {
            bool stream = json_get_bool(req.body, "stream", false);
            if (stream) handle_chat_completions_stream(fd, req.body);
            else send_response(fd, 200, "application/json", handle_chat_completions_sync(req.body));
        }
    } else if (req.method == "POST" && path == "/v1/embeddings") {
        if (!g_embed_handle) {
            send_response(fd, 503, "application/json", make_error_json("server_error", "Embedding model not loaded"));
        } else {
            send_response(fd, 200, "application/json", handle_embeddings(req.body));
        }
    } else {
        send_response(fd, 404, "application/json", make_error_json("not_found", "Endpoint not found"));
    }
    close(fd);
}

static void* client_thread(void* arg) {
    int fd = *static_cast<int*>(arg);
    delete static_cast<int*>(arg);
    handle_request(fd);
    return nullptr;
}

static void signal_handler(int sig) {
    std::cout << "\nShutting down safely..." << std::endl;
    if (g_chat_handle) rkllm_destroy(g_chat_handle);
    if (g_embed_handle) rkllm_destroy(g_embed_handle);
    exit(0);
}

static void sigabrt_handler(int sig) {
    std::cerr << "\nFATAL: SIGABRT (GGML Assertion Failed). Consider using --mock_embed for Embeddings.\n";
    if (g_chat_handle) rkllm_destroy(g_chat_handle);
    if (g_embed_handle) rkllm_destroy(g_embed_handle);
    exit(1);
}

int main(int argc, char** argv) {
    std::string chat_model_path, embed_model_path;
    int max_new_tokens  = 512;
    int max_context_len = 2048;
    int port            = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chat_model" && i + 1 < argc) chat_model_path = argv[++i];
        else if (arg == "--embed_model" && i + 1 < argc) embed_model_path = argv[++i];
        else if (arg == "--chat_name" && i + 1 < argc) g_chat_model_name = argv[++i];
        else if (arg == "--embed_name" && i + 1 < argc) g_embed_model_name = argv[++i];
        else if (arg == "--max_tokens" && i + 1 < argc) max_new_tokens = std::atoi(argv[++i]);
        else if (arg == "--max_context" && i + 1 < argc) max_context_len = std::atoi(argv[++i]);
        else if (arg == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (arg == "--mock_embed") g_mock_embed = true;
    }

    if (chat_model_path.empty() && embed_model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  --chat_model <path>      Path to chat model (e.g. qwen)\n"
                  << "  --embed_model <path>     Path to embedding model (e.g. bge-m3)\n"
                  << "  --chat_name <name>       API Name for chat (default: qwen-chat)\n"
                  << "  --embed_name <name>      API Name for embed (default: text-embedding)\n"
                  << "  --max_tokens <int>       Max new tokens for chat (default: 512)\n"
                  << "  --max_context <int>      Max context len for chat (default: 2048)\n"
                  << "  --port <int>             Server port (default: 8080)\n"
                  << "  --mock_embed             Use synthetic embeddings to bypass GGML asserts\n";
        return 1;
    }

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGABRT, sigabrt_handler);
    signal(SIGPIPE, SIG_IGN);

    if (!chat_model_path.empty()) {
        std::cout << "Loading Chat model: " << chat_model_path << std::endl;
        RKLLMParam param = rkllm_createDefaultParam();
        param.model_path      = chat_model_path.c_str();
        param.top_k           = 1;
        param.top_p           = 0.95f;
        param.temperature     = 0.8f;
        param.repeat_penalty  = 1.1f;
        param.max_new_tokens  = max_new_tokens;
        param.max_context_len = max_context_len;
        param.skip_special_token = true;
        param.extend_param.embed_flash = 1;

        if (rkllm_init(&g_chat_handle, &param, rkllm_callback) != 0) {
            std::cerr << "Failed to init Chat model!\n";
            return 1;
        }
    }

    if (!embed_model_path.empty()) {
        std::cout << "Loading Embedding model: " << embed_model_path 
                  << (g_mock_embed ? " [Mock/Synthetic Mode]" : " [Native Mode]") << std::endl;
        RKLLMParam param = rkllm_createDefaultParam();
        param.model_path      = embed_model_path.c_str();
        param.max_new_tokens  = 0;
        param.skip_special_token = false;
        param.top_k           = 1;
        param.top_p           = 1.0f;
        param.temperature     = 1.0f;
        param.extend_param.embed_flash = 1;

        if (rkllm_init(&g_embed_handle, &param, rkllm_callback) != 0) {
            std::cerr << "Failed to init Embedding model!\n";
            return 1;
        }
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0 || listen(server_fd, 64) < 0) {
        perror("Failed to bind/listen");
        return 1;
    }

    std::cout << "\n========================================\n"
              << "  RKLLM Server\n"
              << "  Listening on http://0.0.0.0:" << port << "\n"
              << "----------------------------------------\n";
    if (g_chat_handle)  std::cout << "  [Chat]  " << g_chat_model_name << "\n";
    if (g_embed_handle) std::cout << "  [Embed] " << g_embed_model_name << "\n";
    std::cout << "========================================\n\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        struct timeval tv { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        int* arg = new int(client_fd);
        pthread_t tid;
        pthread_create(&tid, nullptr, client_thread, arg);
        pthread_detach(tid);
    }

    close(server_fd);
    return 0;
}
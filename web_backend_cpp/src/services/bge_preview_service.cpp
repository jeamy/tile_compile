#include "services/bge_preview_service.hpp"

#include "tile_compile/image/background_extraction.hpp"

#include <fitsio.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;
using tile_compile::Matrix2Df;

namespace tile_compile::web {
namespace {

constexpr int kMaxEdge = 1600;
struct InputProxy {
    Matrix2Df r, g, b;
    std::vector<uint8_t> mask;
    std::string source;
    std::string signature;
};
struct CachedResult {
    std::unordered_map<std::string, std::vector<unsigned char>> images;
    nlohmann::json diagnostics;
};
std::mutex cache_mutex;
std::unordered_map<std::string, InputProxy> input_cache;
std::unordered_map<std::string, CachedResult> result_cache;

std::string fits_error(int status) {
    char text[FLEN_STATUS]{}; fits_get_errstatus(status, text); return text;
}

std::vector<float> read_plane(const fs::path& path, long plane,
                              long& width, long& height, long& planes) {
    fitsfile* file = nullptr; int status = 0;
    if (fits_open_file(&file, path.string().c_str(), READONLY, &status))
        throw std::runtime_error("Cannot open FITS " + path.string() + ": " + fits_error(status));
    int naxis = 0; long axes[3]{1,1,1};
    if (fits_get_img_dim(file, &naxis, &status) || fits_get_img_size(file, 3, axes, &status) || naxis < 2) {
        fits_close_file(file, &status); throw std::runtime_error("Invalid FITS image " + path.string());
    }
    width = axes[0]; height = axes[1]; planes = naxis >= 3 ? axes[2] : 1;
    if (plane > planes) { fits_close_file(file, &status); throw std::runtime_error("Missing FITS plane"); }
    std::vector<float> out(static_cast<size_t>(width * height)); long first[3]{1,1,plane}; int any = 0;
    if (fits_read_pix(file, TFLOAT, first, width * height, nullptr, out.data(), &any, &status)) {
        const auto message = fits_error(status); fits_close_file(file, &status); throw std::runtime_error(message);
    }
    fits_close_file(file, &status); return out;
}

Matrix2Df matrix_from(const std::vector<float>& values, int rows, int cols) {
    Matrix2Df out(rows, cols);
    for (int y=0;y<rows;++y) for (int x=0;x<cols;++x) out(y,x)=values[static_cast<size_t>(y*cols+x)];
    return out;
}

Matrix2Df resize_matrix(const Matrix2Df& in, int rows, int cols, int interpolation) {
    cv::Mat source(in.rows(), in.cols(), CV_32F);
    for (int y=0;y<in.rows();++y) for (int x=0;x<in.cols();++x) source.at<float>(y,x)=in(y,x);
    cv::Mat target; cv::resize(source,target,cv::Size(cols,rows),0,0,interpolation);
    Matrix2Df out(rows,cols);
    for (int y=0;y<rows;++y) for (int x=0;x<cols;++x) out(y,x)=target.at<float>(y,x);
    return out;
}

std::string signature(const fs::path& image, const fs::path& mask) {
    return image.string()+":"+std::to_string(fs::file_size(image))+":"+
        std::to_string(fs::last_write_time(image).time_since_epoch().count())+":"+
        mask.string()+":"+std::to_string(fs::file_size(mask))+":"+
        std::to_string(fs::last_write_time(mask).time_since_epoch().count());
}

InputProxy load_input(const fs::path& run_dir) {
    const fs::path outputs=run_dir/"outputs";
    fs::path image=outputs/"stacked_rgb_solve.fits";
    if (!fs::exists(image)) image=outputs/"stacked_rgb.fits";
    const fs::path mask_path=outputs/"canvas_mask.fits";
    if (!fs::exists(image)) throw std::runtime_error("No pre-BGE RGB artifact found");
    if (!fs::exists(mask_path)) throw std::runtime_error("canvas_mask.fits is required");
    const std::string sig=signature(image,mask_path);
    const std::string key=fs::weakly_canonical(run_dir).string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it=input_cache.find(key); if(it!=input_cache.end()&&it->second.signature==sig)return it->second;
    }
    long w,h,p; auto rv=read_plane(image,1,w,h,p);
    if(p<3) throw std::runtime_error("Pre-BGE RGB artifact has fewer than three planes");
    long gw,gh,gp,bw,bh,bp; auto gv=read_plane(image,2,gw,gh,gp); auto bv=read_plane(image,3,bw,bh,bp);
    if(gw!=w||gh!=h||bw!=w||bh!=h) throw std::runtime_error("Pre-BGE RGB dimensions mismatch");
    long mw,mh,mp; auto mv=read_plane(mask_path,1,mw,mh,mp);
    if(mw!=w||mh!=h) throw std::runtime_error("Canvas mask dimensions mismatch");
    InputProxy proxy{matrix_from(rv,h,w),matrix_from(gv,gh,gw),matrix_from(bv,bh,bw),{},image.filename().string(),sig};
    proxy.mask.resize(mv.size()); for(size_t i=0;i<mv.size();++i)proxy.mask[i]=std::isfinite(mv[i])&&mv[i]>0.5f?1u:0u;
    const int edge=std::max<int>(h,w);
    if(edge>kMaxEdge){
        const double s=static_cast<double>(kMaxEdge)/edge; const int rows=std::max(1,static_cast<int>(std::lround(h*s))); const int cols=std::max(1,static_cast<int>(std::lround(w*s)));
        proxy.r=resize_matrix(proxy.r,rows,cols,cv::INTER_AREA); proxy.g=resize_matrix(proxy.g,rows,cols,cv::INTER_AREA); proxy.b=resize_matrix(proxy.b,rows,cols,cv::INTER_AREA);
        cv::Mat m(h,w,CV_8U,proxy.mask.data()), mr; cv::resize(m,mr,cv::Size(cols,rows),0,0,cv::INTER_NEAREST);
        proxy.mask.assign(mr.data,mr.data+static_cast<size_t>(rows*cols));
    }
    std::lock_guard<std::mutex> lock(cache_mutex); if(input_cache.size()>=4)input_cache.erase(input_cache.begin()); input_cache[key]=proxy; return proxy;
}

template<class T>T val(const nlohmann::json& p,const char* key,T fallback){return p.contains(key)?p.at(key).get<T>():fallback;}

image::BGEConfig make_config(const nlohmann::json& p,const InputProxy& input,const nlohmann::json& polygons){
    image::BGEConfig cfg; cfg.enabled=true; cfg.method="autobge";
    cfg.autobge.num_sample_points=val<int>(p,"num_sample_points",0);
    cfg.autobge.poly_degree=val<int>(p,"poly_degree",2); cfg.autobge.rbf_smooth=val<float>(p,"rbf_smooth",2.f);
    cfg.autobge.downsample_scale=val<int>(p,"downsample_scale",4); cfg.autobge.patch_size=val<int>(p,"patch_size",35);
    cfg.autobge.patch_estimator=val<std::string>(p,"patch_estimator","sigma_clipped_median");
    cfg.autobge.stretch_mode=val<std::string>(p,"stretch_mode","linear"); cfg.autobge.stretch_target_median=val<float>(p,"stretch_target_median",.25f);
    cfg.autobge.border_margin=val<int>(p,"border_margin",10); cfg.autobge.bright_exclusion_fraction=val<float>(p,"bright_exclusion_fraction",.2f);
    cfg.autobge.gradient_descent_max_iters=val<int>(p,"gradient_descent_max_iters",100); cfg.autobge.random_seed=val<int>(p,"random_seed",42);
    cfg.autobge.normalize_between_stages=val<bool>(p,"normalize_between_stages",true); cfg.autobge.apply_guards=val<bool>(p,"apply_guards",true);
    cfg.autobge.mono_mode=val<std::string>(p,"mono_mode","rgb_duplicate");
    std::vector<std::string> bad; auto range=[&](const char*n,double v,double lo,double hi){if(!std::isfinite(v)||v<lo||v>hi){std::ostringstream s;s<<n<<'='<<v<<" (expected "<<lo<<".."<<hi<<')';bad.push_back(s.str());}};
    range("num_sample_points",cfg.autobge.num_sample_points,0,3000); range("poly_degree",cfg.autobge.poly_degree,1,6); range("rbf_smooth",cfg.autobge.rbf_smooth,0,10); range("downsample_scale",cfg.autobge.downsample_scale,1,8); range("patch_size",cfg.autobge.patch_size,3,101); range("stretch_target_median",cfg.autobge.stretch_target_median,.01,.99); range("border_margin",cfg.autobge.border_margin,0,250); range("bright_exclusion_fraction",cfg.autobge.bright_exclusion_fraction,.01,.99); range("gradient_descent_max_iters",cfg.autobge.gradient_descent_max_iters,1,500);
    if((cfg.autobge.patch_size%2)==0)bad.push_back("patch_size must be odd");
    if(cfg.autobge.patch_estimator!="median"&&cfg.autobge.patch_estimator!="sigma_clipped_median")bad.push_back("patch_estimator is invalid");
    if(cfg.autobge.stretch_mode!="none"&&cfg.autobge.stretch_mode!="linear"&&cfg.autobge.stretch_mode!="mtf")bad.push_back("stretch_mode is invalid");
    if(cfg.autobge.mono_mode!="rgb_duplicate"&&cfg.autobge.mono_mode!="disabled")bad.push_back("mono_mode is invalid");
    if(!bad.empty()){std::ostringstream s;s<<"Invalid BGE parameter values: ";for(size_t i=0;i<bad.size();++i){if(i)s<<"; ";s<<bad[i];}throw std::invalid_argument(s.str());}
    cfg.common_valid_mask=input.mask; cfg.common_mask_rows=input.r.rows(); cfg.common_mask_cols=input.r.cols();
    if(polygons.is_array()&&!polygons.empty()){
        cfg.sampling_valid_mask.assign(input.mask.size(),1u); cfg.sampling_mask_rows=input.r.rows(); cfg.sampling_mask_cols=input.r.cols();
        for(const auto& poly:polygons){if(!poly.is_array()||poly.size()<3)throw std::invalid_argument("Exclusion polygon requires at least 3 points");
            for(int y=0;y<input.r.rows();++y)for(int x=0;x<input.r.cols();++x){const double px=(x+.5)/input.r.cols(),py=(y+.5)/input.r.rows();bool inside=false;for(size_t i=0,j=poly.size()-1;i<poly.size();j=i++){
                const double xi=poly[i][0].get<double>(),yi=poly[i][1].get<double>(),xj=poly[j][0].get<double>(),yj=poly[j][1].get<double>(); if(xi<0||xi>1||yi<0||yi>1)throw std::invalid_argument("Exclusion polygon coordinates must be in [0,1]"); if(((yi>py)!=(yj>py))&&px<(xj-xi)*(py-yi)/((yj-yi)+1e-20)+xi)inside=!inside;}
                if(inside)cfg.sampling_valid_mask[static_cast<size_t>(y*input.r.cols()+x)]=0u;}}
    }
    return cfg;
}

std::pair<float,float> display_range(const InputProxy& in){std::vector<float> v;v.reserve(in.mask.size()/8+1);for(int y=0;y<in.r.rows();++y)for(int x=0;x<in.r.cols();++x){size_t i=static_cast<size_t>(y*in.r.cols()+x);if(in.mask[i]&&i%8==0){v.push_back(in.r(y,x));v.push_back(in.g(y,x));v.push_back(in.b(y,x));}}if(v.empty())return{0,1};std::sort(v.begin(),v.end());return{v[v.size()/1000],v[std::min(v.size()-1,v.size()*999/1000)]};}

std::vector<unsigned char> encode(const Matrix2Df&r,const Matrix2Df&g,const Matrix2Df&b,float lo,float hi){cv::Mat out(r.rows(),r.cols(),CV_8UC3);const float d=std::max(hi-lo,1e-9f);for(int y=0;y<r.rows();++y)for(int x=0;x<r.cols();++x){auto&p=out.at<cv::Vec3b>(y,x);p[2]=cv::saturate_cast<uchar>((r(y,x)-lo)/d*255);p[1]=cv::saturate_cast<uchar>((g(y,x)-lo)/d*255);p[0]=cv::saturate_cast<uchar>((b(y,x)-lo)/d*255);}std::vector<uchar> png;if(!cv::imencode(".png",out,png))throw std::runtime_error("PNG encoding failed");return png;}

} // namespace

BgePreviewResult create_bge_preview(const fs::path& run_dir,const nlohmann::json& params,const nlohmann::json& polygons,const nlohmann::json& manual_sample_points,const std::string& view){
    BgePreviewResult out;
    try{
        InputProxy input=load_input(run_dir); const std::string cache_key=input.signature+params.dump()+polygons.dump()+manual_sample_points.dump();
        {std::lock_guard<std::mutex>lock(cache_mutex);auto it=result_cache.find(cache_key);if(it!=result_cache.end()){out.ok=true;out.status=200;out.diagnostics=it->second.diagnostics;if(view!="diagnostics"){auto im=it->second.images.find(view);if(im==it->second.images.end())throw std::invalid_argument("Unknown BGE preview view");out.png=im->second;}return out;}}
        auto cfg=make_config(params,input,polygons);if(manual_sample_points.is_array()){for(const auto& pt:manual_sample_points){if(!pt.is_array()||pt.size()<2)continue;const double nx=pt[0].get<double>(), ny=pt[1].get<double>();if(nx<0.0||nx>1.0||ny<0.0||ny>1.0)continue;cfg.autobge.user_sample_points.push_back({static_cast<float>(nx),static_cast<float>(ny)});}} auto models=image::build_autobge_models(input.r,input.g,input.b,cfg); if(!models.success)throw std::runtime_error("AutoBGE could not build channel models");
        Matrix2Df cr=input.r,cg=input.g,cb=input.b; image::BGEDiagnostics diag; diag.attempted=true;diag.bge_method="autobge";diag.method="autobge";diag.channels=models.channel_diagnostics;
        const bool applied=image::finalize_bge_from_channel_models(cr,cg,cb,models.channel_models,models.channel_diagnostics,cfg,&diag);
        Matrix2Df br=models.channel_models[0].model,bg=models.channel_models[1].model,bb=models.channel_models[2].model;
        auto range=display_range(input); std::vector<float> bv;for(int y=0;y<br.rows();y+=4)for(int x=0;x<br.cols();x+=4){bv.push_back(br(y,x));bv.push_back(bg(y,x));bv.push_back(bb(y,x));}std::sort(bv.begin(),bv.end());float blo=bv.empty()?0:bv[bv.size()/1000],bhi=bv.empty()?1:bv[std::min(bv.size()-1,bv.size()*999/1000)];
        CachedResult cached;cached.images["original"]=encode(input.r,input.g,input.b,range.first,range.second);cached.images["corrected"]=encode(cr,cg,cb,range.first,range.second);cached.images["background"]=encode(br,bg,bb,blo,bhi);
        nlohmann::json channels=nlohmann::json::array(),points=nlohmann::json::array();const auto& reported_channels=diag.channels.empty()?models.channel_diagnostics:diag.channels;for(const auto&ch:reported_channels){channels.push_back({{"channel",ch.channel_name},{"fit_rms",ch.fit_rms_residual},{"samples",ch.tile_samples_valid},{"guard_rejected",ch.guard_rejected},{"guard_reason",ch.guard_reason},{"flat_pre",ch.guard_flat_pre},{"flat_post",ch.guard_flat_post},{"slope_pre",ch.guard_slope_pre},{"slope_post",ch.guard_slope_post}});for(const auto&p:ch.grid_cells)if(points.size()<3000)points.push_back({{"x",p.center_x/input.r.cols()},{"y",p.center_y/input.r.rows()},{"channel",ch.channel_name}});}
        cached.diagnostics={{"success",applied},{"source",input.source},{"width",input.r.cols()},{"height",input.r.rows()},{"failure_reason",diag.failure_reason},{"channels",channels},{"sample_points",points}};
        {std::lock_guard<std::mutex>lock(cache_mutex);if(result_cache.size()>=4)result_cache.erase(result_cache.begin());result_cache[cache_key]=cached;}
        out.ok=true;out.status=200;out.diagnostics=cached.diagnostics;if(view!="diagnostics"){auto im=cached.images.find(view);if(im==cached.images.end())throw std::invalid_argument("Unknown BGE preview view");out.png=im->second;}
    }catch(const std::invalid_argument&e){out.status=400;out.error=e.what();}catch(const std::exception&e){out.status=400;out.error=e.what();}return out;
}

} // namespace tile_compile::web

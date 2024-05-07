use js_sys::{Array, Uint32Array};
use wasm_bindgen::prelude::*;
use web_sys::{Element, WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlTexture};

use naga::{
    back::glsl::{Options, PipelineOptions, Version, Writer},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo,
};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

pub fn parse_and_validate_wgsl(
    src: &str,
) -> Result<(naga::Module, ModuleInfo), Box<dyn std::error::Error>> {
    let mut frontend = naga::front::wgsl::Frontend::new();

    let module = frontend.parse(src)?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator.validate(&module)?;

    Ok((module, info))
}

#[wasm_bindgen(start)]
fn start() -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    let vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r##"#version 300 es
        precision highp float;

        in vec4 position;
        in vec2 texcoords;
        out vec2 thread_uv;

        void main() {
            thread_uv = texcoords;
            gl_Position = position;
        }
        "##,
    )?;

    let src = "
            @group(0)
            @binding(0)
            var<storage, read> x: array<f32>;

            @group(0)
            @binding(1)
            var<storage, read_write> out: array<f32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }

                var counter = 0.0;
                for (var i = 0; i < 10; i++) {
                    counter += 1.0;
                }

                // if out is used on the right side: problem at the moment
                out[global_id.x] = counter * x[global_id.x];
                // out[global_id.x] = 3.0;
            }

    ";

    let (module, info) = parse_and_validate_wgsl(&src).unwrap();

    // 310 is required for compute shaders
    let version = Version::Embedded {
        version: 310,
        is_webgl: true,
    };
    let mut glsl = String::new();
    let options = Options {
        version,
        ..Default::default()
    };
    let pipeline_options = PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };

    let mut writer = Writer::new(
        &mut glsl,
        &module,
        &info,
        &options,
        &pipeline_options,
        BoundsCheckPolicies::default(),
    )
    .unwrap();

    let reflection_info = writer.write_webgl_compute().unwrap();
    log!("ref info: {reflection_info:?}");

    let output_storage_layout_names = reflection_info.outputs.values().collect::<Vec<_>>();

    let input_storage_uniform_names = reflection_info
        .input_storage_uniforms
        .values()
        .collect::<Vec<_>>();

    let glsl = glsl.replace("#version 310 es", "#version 300 es");
    log!("glsl: {glsl}");

    let frag_shader = compile_shader(&context, WebGl2RenderingContext::FRAGMENT_SHADER, &glsl)?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));

    let frame_buf = context
        .create_framebuffer()
        .ok_or("Failed to create frame buffer")?;

    #[rustfmt::skip]
    let vertices: [f32; 12] = [
        -1.0,-1.0, 0.0, 
        -1.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 
         1.0,-1.0, 0.0
     ];

    let position_attribute_location = context.get_attrib_location(&program, "position");
    let position_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&position_buffer));

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&vertices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    // let vao = context
    //     .create_vertex_array()
    //     .ok_or("Could not create vertex array object")?;
    // context.bind_vertex_array(Some(&vao));

    context.enable_vertex_attrib_array(position_attribute_location as u32);
    context.vertex_attrib_pointer_with_i32(
        position_attribute_location as u32,
        3,
        WebGl2RenderingContext::FLOAT,
        false,
        0,
        0,
    );

    // context.bind_vertex_array(Some(&vao));

    #[rustfmt::skip]
    let tex_coords: [f32; 8] = [
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0
    ];

    let texcoords_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    let texcoords_attribute_location = context.get_attrib_location(&program, "texcoords");
    context.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&texcoords_buffer),
    );
    context.enable_vertex_attrib_array(texcoords_attribute_location as u32);
    context.vertex_attrib_pointer_with_i32(
        texcoords_attribute_location as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        0,
        0,
    );

    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&tex_coords);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

    let indices_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&indices_buffer),
    );

    unsafe {
        let positions_array_buf_view = js_sys::Uint16Array::view(&indices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let thread_viewport_width_uniform = context
        .get_uniform_location(&program, "thread_viewport_width")
        .ok_or("cannot find thread vpw")?;
    let thread_viewport_height_uniform = context
        .get_uniform_location(&program, "thread_viewport_height")
        .ok_or("cannot find thread vpw")?;

    // do not bubble up error -> it is possible that the internal glsl compiler removes unused uniforms
    let gws_x_uniform = context
        .get_uniform_location(&program, "gws_x");
    let gws_y_uniform = context
        .get_uniform_location(&program, "gws_y");
    let gws_z_uniform = context
        .get_uniform_location(&program, "gws_z");

    let mut input_uniforms = Vec::with_capacity(input_storage_uniform_names.len());

    // TODO: support e.g. floats as inputs
    for uniform_name in input_storage_uniform_names {
        input_uniforms.push([
            context
                .get_uniform_location(&program, uniform_name)
                .ok_or("cannot find uniform input")?,
            context
                .get_uniform_location(&program, &format!("{uniform_name}_texture_width"))
                .ok_or("cannot find uniform input width")?,
            context
                .get_uniform_location(&program, &format!("{uniform_name}_texture_height"))
                .ok_or("cannot find uniform input height")?,
        ]);
    }

    context
        .get_uniform_location(&program, "_group_0_binding_0_cs_texture_height")
        .ok_or("not found")?;

    let mut output_size_uniforms = Vec::with_capacity(output_storage_layout_names.len());

    for uniform_name in output_storage_layout_names {
        output_size_uniforms.push([
            context
                .get_uniform_location(&program, &format!("{uniform_name}_texture_width"))
                .ok_or("cannot find uniform out width")?,
            context
                .get_uniform_location(&program, &format!("{uniform_name}_texture_height"))
                .ok_or("cannot find uniform out height")?,
        ]);
    }

    let mut lhs = WebGlBuffer::new(&context, 16).unwrap();

    let f32_slice = unsafe {
        std::slice::from_raw_parts_mut(
            lhs.texture_data.as_mut_ptr() as *mut f32,
            lhs.texture_data.len() / 4,
        )
    };
    for (idx, x) in f32_slice.iter_mut().enumerate() {
        *x = idx as f32;
    }

    let lhs = lhs.push().unwrap();

    /*let mut rhs = WebGlBuffer::new(&context, 16).unwrap();

    let f32_slice = unsafe { std::slice::from_raw_parts_mut(rhs.texture_data.as_mut_ptr() as *mut f32, rhs.texture_data.len() / 4) };
    for x in f32_slice {
        *x = 3.;
    }*/

    let mut out = WebGlBuffer::new(&context, 16).unwrap();

    // let color_attachments = Array::new();
    let color_attachments = Uint32Array::new(&JsValue::from(1));

    let attachment = WebGl2RenderingContext::COLOR_ATTACHMENT0 + 0;
    color_attachments.set_index(0, attachment);
    // color_attachments.push(&JsValue::from(attachment));

    context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&frame_buf));
    context.draw_buffers(&color_attachments);

    context.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0 + 0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(&out.texture),
        0,
    );
    assert_eq!(
        context.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER),
        WebGl2RenderingContext::FRAMEBUFFER_COMPLETE
    );

    context.use_program(Some(&program));

    context.viewport(0, 0, out.texture_width as i32, out.texture_height as i32);
    context.uniform1ui(
        Some(&thread_viewport_width_uniform),
        out.texture_width as u32,
    );
    context.uniform1ui(
        Some(&thread_viewport_height_uniform),
        out.texture_height as u32,
    );

    // context.uniform1ui(Some(&gws_x_uniform), out.texture_width as u32);
    // context.uniform1ui(Some(&gws_y_uniform), out.texture_height as u32);
    // context.uniform1ui(Some(&gws_z_uniform), 1);
    context.uniform1ui(
        gws_x_uniform.as_ref(),
        out.texture_width as u32 * out.texture_height as u32,
    );
    context.uniform1ui(gws_y_uniform.as_ref(), 1);
    context.uniform1ui(gws_z_uniform.as_ref(), 1);

    for (idx, (input_uniform, gl_buf)) in input_uniforms.iter().zip([&lhs]).enumerate() {
        context.uniform1i(Some(&input_uniform[0]), idx as i32);
        context.uniform1ui(Some(&input_uniform[1]), gl_buf.texture_width as u32);
        context.uniform1ui(Some(&input_uniform[2]), gl_buf.texture_height as u32);
        context.active_texture(WebGl2RenderingContext::TEXTURE0 + idx as u32);
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&gl_buf.texture))
    }

    for (_idx, (output_size_uniform, gl_buf)) in output_size_uniforms.iter().zip([&out]).enumerate()
    {
        context.uniform1ui(Some(&output_size_uniform[0]), gl_buf.texture_width as u32);
        context.uniform1ui(Some(&output_size_uniform[1]), gl_buf.texture_height as u32);
    }

    // let vert_count = (vertices.len() / 3) as i32;
    // draw(&context, vert_count);

    context.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&indices_buffer),
    );

    context.draw_elements_with_i32(
        WebGl2RenderingContext::TRIANGLES,
        6,
        WebGl2RenderingContext::UNSIGNED_SHORT,
        0,
    );

    // for all outputs (mind + 0)
    context.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0 + 0,
        WebGl2RenderingContext::TEXTURE_2D,
        None,
        0,
    );

    context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);

    // read .. bind again

    context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&frame_buf));
    context.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0 + 0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(&out.texture),
        0,
    );

    assert_eq!(
        context.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER),
        WebGl2RenderingContext::FRAMEBUFFER_COMPLETE
    );

    context
        .read_pixels_with_u8_array_and_dst_offset(
            0,
            0,
            out.texture_width as i32,
            out.texture_height as i32,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            &mut out.texture_data,
            0,
        )
        .unwrap();

    context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);

    log!("out: {out:?}");

    let f32_slice = unsafe {
        std::slice::from_raw_parts_mut(
            out.texture_data.as_mut_ptr() as *mut f32,
            out.texture_data.len() / 4,
        )
    };

    log!("out: {f32_slice:?}");

    Ok(())
}

fn compute_texture_dimensions(length: usize) -> (usize, usize) {
    let sqrt = (length as f64).sqrt().ceil();
    (sqrt as usize, sqrt as usize)
}

#[derive(Debug)]
struct WebGlBuffer<'a> {
    texture: WebGlTexture,
    texture_data: Vec<u8>,
    texture_width: usize,
    texture_height: usize,
    context: &'a WebGl2RenderingContext,
}

impl<'a> WebGlBuffer<'a> {
    pub fn new(context: &'a WebGl2RenderingContext, len: usize) -> Option<Self> {
        let texture = context.create_texture()?;
        let texture_data = vec![0u8; len * 4];
        let (texture_width, texture_height) = compute_texture_dimensions(len);

        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);

        Some(
            WebGlBuffer {
                texture,
                texture_data,
                texture_width,
                texture_height,
                context,
            }
            .push()?,
        )
    }

    pub fn push(self) -> Option<Self> {
        self.context
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.texture));

        unsafe {
            let texture_data = js_sys::Uint8Array::view(&self.texture_data);

            self.context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA as i32,
                self.texture_width as i32,
                self.texture_height as i32,
                0,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&texture_data)
            ).ok()?;
        }
        self.context
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);
        Some(self)
    }
}

pub fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub fn link_program(
    context: &WebGl2RenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}
